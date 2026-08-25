from copy import deepcopy
from enum import IntEnum

import numpy as np

from gym import spaces
from gym_minigrid.minigrid import COLOR_NAMES, DIR_TO_VEC, OBJECT_TO_IDX, STATE_TO_IDX, TILE_PIXELS, Door, Goal, Grid, MiniGridEnv, Wall
from gym_minigrid.envs.multiroom import Room

# from . import register
from envs.registration import register as gym_register


""" Based on MultiRoomEnv in MiniGrid """
class MultiRoomEnv(MiniGridEnv):

    def __init__(
            self,
            min_num_rooms,
            max_num_rooms,
            min_room_size=4,
            max_room_size=10,
            grid_size=25,
            max_steps=None,
            seed=None,
            include_mission_in_obs=False,
        ):

        assert isinstance(min_num_rooms, int), \
            f"Argument \"min_num_rooms\" must be an integer, but it is {type(min_num_rooms)}."
        assert min_num_rooms > 0, \
            f"Argument \"min_num_rooms\" must be greater than zero, but it is {min_num_rooms}."

        assert isinstance(max_num_rooms, int), \
            f"Argument \"max_num_rooms\" must be an integer, but it is {type(max_num_rooms)}."
        assert max_num_rooms >= min_num_rooms, \
            f"Argument \"max_num_rooms\" must be equal to or greater than \"min_num_rooms\" ({min_num_rooms}), but it is {max_num_rooms}."

        assert isinstance(min_room_size, int), \
            f"Argument \"min_room_size\" must be an integer, but it is {type(min_room_size)}."
        assert min_room_size >= 4, \
            f"Argument \"min_room_size\" must be equal to or greater than 4, but it is {min_room_size}."

        assert isinstance(max_room_size, int), \
            f"Argument \"max_room_size\" must be an integer, but it is {type(max_room_size)}."
        assert max_room_size >= min_room_size, \
            f"Argument \"max_room_size\" must be equal to or greater than \"min_room_size\" ({min_room_size}), but it is {max_room_size}."

        self.min_num_rooms = min_num_rooms
        self.max_num_rooms = max_num_rooms
        self.min_room_size = min_room_size
        self.max_room_size = max_room_size

        assert isinstance(grid_size, int), \
            f"Argument \"grid_size\" must be an integer, but it is {type(grid_size)}."

        self.grid_size = grid_size

        self.max_steps = max_steps if max_steps is not None else self._default_max_steps(max_num_rooms)

        kwargs = {"seed": seed} if seed is not None else {}

        self.include_mission_in_obs = include_mission_in_obs

        self.rooms = []
        super().__init__(
            grid_size=grid_size,
            max_steps=self.max_steps,
            **kwargs,
        )

        # The original minigrid observation space does not include direction or mission, so let's add them
        assert len(self.observation_space.spaces) == 1 and 'image' in self.observation_space.spaces
        obs_spaces = {
            'image': self.observation_space['image'],
            'direction': spaces.Box(low=0, high=3, shape=(1,), dtype='uint8'), #spaces.Discrete(len(DIR_TO_VEC)),
        }
        if self.include_mission_in_obs:
            obs_spaces['mission'] = spaces.Space(dtype=str)

        self.observation_space = spaces.Dict(obs_spaces)

        self.passable = True

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

    def seed(self, seed):
        returned_seed = super().seed(seed=seed)
        assert returned_seed == [seed]
        self.seed_value = seed
        return returned_seed

    @property
    def encoding(self):
        encoded_grid = self.grid.encode()
        # this env has no objects to pick up, so don't worry about this case
        assert self.carrying is None
        if self.agent_pos is not None and self.agent_dir is not None:
            if encoded_grid[self.agent_pos[0], self.agent_pos[1], 0] == OBJECT_TO_IDX['empty']:
                # no object at agent's position: encode agent onto the grid
                assert encoded_grid[self.agent_pos[0], self.agent_pos[1], 1] == 0
                assert encoded_grid[self.agent_pos[0], self.agent_pos[1], 2] == 0
                encoded_grid[self.agent_pos[0], self.agent_pos[1], 0] = OBJECT_TO_IDX['agent']
                encoded_grid[self.agent_pos[0], self.agent_pos[1], 2] = self.agent_dir
            elif encoded_grid[self.agent_pos[0], self.agent_pos[1], 0] == OBJECT_TO_IDX['door']:
                # confirm that agent is on an open door
                assert encoded_grid[self.agent_pos[0], self.agent_pos[1], 2] == 0
                assert len(STATE_TO_IDX) < 10
                agent_state = (self.agent_dir + 1) * 10 # the + 1 is to disambiguate if the agent is facing right
                encoded_grid[self.agent_pos[0], self.agent_pos[1], 2] = agent_state
            else:
                raise NotImplementedError

        return encoded_grid

    @staticmethod
    def room_info_from_encoding_and_position(encoded_level, position):
        width, height = encoded_level.shape[:2]
        pos_x, pos_y = position

        # get left x
        room_left_x = None
        for i in range(pos_x - 1, -1, -1):
            if encoded_level[i, pos_y, 0] in [OBJECT_TO_IDX['wall'], OBJECT_TO_IDX['door']]:
                room_left_x = i
                break
        assert room_left_x is not None

        # get up y
        room_top_y = None
        for j in range(pos_y - 1, -1, -1):
            if encoded_level[pos_x, j, 0] in [OBJECT_TO_IDX['wall'], OBJECT_TO_IDX['door']]:
                room_top_y = j
                break
        assert room_top_y is not None

        # get right x
        room_right_x = None
        for i in range(pos_x + 1, width):
            if encoded_level[i, pos_y, 0] in [OBJECT_TO_IDX['wall'], OBJECT_TO_IDX['door']]:
                room_right_x = i
                break
        assert room_right_x is not None

        # get bottom y
        room_bottom_y = None
        for j in range(pos_y + 1, height):
            if encoded_level[pos_x, j, 0] in [OBJECT_TO_IDX['wall'], OBJECT_TO_IDX['door']]:
                room_bottom_y = j
                break
        assert room_bottom_y is not None

        size_x = room_right_x - room_left_x + 1
        size_y = room_bottom_y - room_top_y + 1

        # find any doors
        door_positions = []
        door_walls = []
        for i in range(room_left_x, room_left_x + size_x):
            for j in range(room_top_y, room_top_y + size_y):
                if encoded_level[i, j, 0] == OBJECT_TO_IDX['door']:
                    # Door on the right
                    if i == room_right_x:
                        assert j != room_top_y and j != room_bottom_y
                        door_wall = 0
                    # Door on the south
                    elif j == room_bottom_y:
                        assert i != room_left_x and i != room_right_x
                        door_wall = 1
                    # Door on the left
                    elif i == room_left_x:
                        assert j != room_top_y and j != room_bottom_y
                        door_wall = 2
                    # Door on the top
                    elif j == room_top_y:
                        assert i != room_left_x and i != room_right_x
                        door_wall = 3
                    else:
                        # Door in the middle of the room -- this should not happen
                        raise ValueError

                    door_positions.append((i, j))
                    door_walls.append(door_wall)

        top = (room_left_x, room_top_y)
        size = (size_x, size_y)

        return top, size, door_positions, door_walls

    def rooms_from_encoding_and_position_in_last_room(self, encoded_level, pos_in_last_room):
        # Determine list of rooms, starting from the last room and working backward
        room_top, room_size, door_positions, door_walls = self.room_info_from_encoding_and_position(encoded_level, pos_in_last_room)

        assert len(door_positions) == 0 or len(door_positions) == 1

        if len(door_positions) == 0:
            # No more rooms
            room = Room(room_top, room_size, room_top, None)
            last_room_entry_door_wall = None
            first_room_found = True
        else:
            room = Room(room_top, room_size, door_positions[0], None)
            last_room_entry_door_wall = door_walls[0]
            first_room_found = False
        room_list = [room]

        while not first_room_found:
            last_room_door_pos = room_list[-1].entryDoorPos
            if last_room_entry_door_wall == 0:
                next_room_pos = (last_room_door_pos[0] + 1, last_room_door_pos[1])
            elif last_room_entry_door_wall == 1:
                next_room_pos = (last_room_door_pos[0], last_room_door_pos[1] + 1)
            elif last_room_entry_door_wall == 2:
                next_room_pos = (last_room_door_pos[0] - 1, last_room_door_pos[1])
            elif last_room_entry_door_wall == 3:
                next_room_pos = (last_room_door_pos[0], last_room_door_pos[1] - 1)
            else:
                raise ValueError

            next_room_top, next_room_size, next_door_positions, next_door_walls = self.room_info_from_encoding_and_position(encoded_level, next_room_pos)

            if len(next_door_positions) == 1:
                assert len(next_door_walls) == 1

                next_room_entry_door_pos = next_room_top
                next_room_exit_door_pos = next_door_positions[0]

                last_room_entry_door_wall = None
                first_room_found = True

            elif len(next_door_positions) == 2:
                assert len(next_door_walls) == 2

                x_exit_door = next_door_positions.index(last_room_door_pos)
                indices = [0,1]
                indices.remove(x_exit_door)
                x_entry_door = indices[0]

                next_room_entry_door_pos = next_door_positions[x_entry_door]
                next_room_exit_door_pos = next_door_positions[x_exit_door]

                last_room_entry_door_wall = next_door_walls[x_entry_door]

            else:
                raise ValueError

            next_room = Room(
                next_room_top,
                next_room_size,
                next_room_entry_door_pos,
                next_room_exit_door_pos,
            )
            room_list.append(next_room)

        room_list.reverse()

        return room_list

    def get_complexity_info(self):
        avg_eq_room_size = np.mean([np.sqrt(np.prod(r.size)) for r in self.rooms]) if len(self.rooms) > 0 else 0
        room_sizes = [r.size for r in self.rooms]
        complexity_info = {
            'num_rooms': len(self.rooms),
            'avg_eq_room_size': avg_eq_room_size,
            'room_sizes': room_sizes,
        }

        return complexity_info

    @staticmethod
    def _default_max_steps(max_num_rooms):
        return max_num_rooms * 20

    """ Based on minigrid __str__, but supporting no agents"""
    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            'wall'          : 'W',
            'floor'         : 'F',
            'door'          : 'D',
            'key'           : 'K',
            'ball'          : 'A',
            'box'           : 'B',
            'goal'          : 'G',
            'lava'          : 'V',
        }

        # Short string for opened door
        OPENDED_DOOR_IDS = '_'

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {
            0: '>',
            1: 'V',
            2: '<',
            3: '^'
        }

        str = ''

        for j in range(self.grid.height):

            for i in range(self.grid.width):

                if self.agent_pos is not None and i == self.agent_pos[0] and j == self.agent_pos[1]:
                    str += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    continue

                c = self.grid.get(i, j)

                if c == None:
                    str += '  '
                    continue

                if c.type == 'door':
                    if c.is_open:
                        str += '__'
                    elif c.is_locked:
                        str += 'L' + c.color[0].upper()
                    else:
                        str += 'D' + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += '\n'

        return str

    """ Based on minigrid render, but supporting no agents"""
    def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        highlight_mask = None
        if self.agent_pos is not None and self.agent_dir is not None:
            # Compute which cells are visible to the agent
            _, vis_mask = self.gen_obs_grid()

            # Compute the world coordinates of the bottom-left corner
            # of the agent's view area
            f_vec = self.dir_vec
            r_vec = self.right_vec
            top_left = self.agent_pos + f_vec * (self.agent_view_size-1) - r_vec * (self.agent_view_size // 2)

            # Mask of which cells to highlight
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

            # For each cell in the visibility mask
            for vis_j in range(0, self.agent_view_size):
                for vis_i in range(0, self.agent_view_size):
                    # If this cell is not visible, don't highlight it
                    if not vis_mask[vis_i, vis_j]:
                        continue

                    # Compute the world coordinates of this cell
                    abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                    if abs_i < 0 or abs_i >= self.width:
                        continue
                    if abs_j < 0 or abs_j >= self.height:
                        continue

                    # Mark this cell to be highlighted
                    highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None
        )

        if mode == 'human':
            self.window.show_img(img)
            self.window.set_caption(self.mission)

        return img

    def gen_obs(self):
        obs = super().gen_obs()
        if not self.include_mission_in_obs:
            del obs['mission']
        # Vectorize direction to retain compatibility with MultiGrid
        obs['direction'] = [obs['direction']]
        return obs

    def _gen_grid(self, width, height):

        self._set_rooms(width, height)

        self._set_agent()

        self._set_goal_position()

        self._set_mission()

    def _set_rooms(self, width, height, num_rooms=None):
        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.min_num_rooms, self.max_num_rooms+1)
        if num_rooms is not None:
            numRooms = num_rooms

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (
                self._rand_int(0, width - 2),
                self._rand_int(0, width - 2)
            )

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=self.min_room_size, #4,
                maxSz=self.max_room_size,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):

            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColors = set(COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                doorColor = self._rand_elem(sorted(doorColors))

                entryDoor = Door(doorColor)
                self.grid.set(*room.entryDoorPos, entryDoor)
                prevDoorColor = doorColor

                prevRoom = roomList[idx-1]
                prevRoom.exitDoorPos = room.entryDoorPos

    def _set_agent(self):
        assert len(self.rooms) > 0
        # Randomize the starting agent position and direction
        self.place_agent(self.rooms[0].top, self.rooms[0].size)

    def _set_goal_position(self):
        assert len(self.rooms) > 0
        # Place the final goal in the last room
        self.goal_pos = self.place_obj(Goal(), self.rooms[-1].top, self.rooms[-1].size)

    def _set_mission(self):
        self.mission = 'traverse the rooms to get to the goal'

    def _placeRoom(
        self,
        numLeft,
        roomList,
        minSz,
        maxSz,
        entryDoorWall,
        entryDoorPos
    ):
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz+1)
        sizeY = self._rand_int(minSz, maxSz+1)

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.width or topY + sizeY >= self.height:
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = \
                topX + sizeX < room.top[0] or \
                room.top[0] + room.size[0] <= topX or \
                topY + sizeY < room.top[1] or \
                room.top[1] + room.size[1] <= topY

            if not nonOverlap:
                return False

        # Add this room to the list
        roomList.append(Room(
            (topX, topY),
            (sizeX, sizeY),
            entryDoorPos,
            None
        ))

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        for i in range(0, 8):

            # Pick which wall to place the out door on
            wallSet = set((0, 1, 2, 3))
            wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exitDoorWall == 0:
                exitDoorPos = (
                    topX + sizeX - 1,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on south wall
            elif exitDoorWall == 1:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY + sizeY - 1
                )
            # Exit on left wall
            elif exitDoorWall == 2:
                exitDoorPos = (
                    topX,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on north wall
            elif exitDoorWall == 3:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY
                )
            else:
                assert False

            # Recursively create the other rooms
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos
            )

            if success:
                break

        return True

    """ consistency checker """
    def _test_initial_validity(self):

        assert self.step_count == 0

        # Rooms exist
        num_rooms = len(self.rooms)
        if num_rooms == 0:
            return False

        # Goal exists
        if self.goal_pos is None:
            return False

        # Goal position is consistent with grid
        if self.grid.get(*self.goal_pos).type != 'goal':
            return False

        # Rooms should be consistent when obtained from encoding
        rooms_from_grid = self.rooms_from_encoding_and_position_in_last_room(self.encoding, self.goal_pos)

        if num_rooms != len(rooms_from_grid):
            return False

        for self_room, room_from_grid in zip(self.rooms, rooms_from_grid):
            if self_room.top != room_from_grid.top:
                return False
            if self_room.size != room_from_grid.size:
                return False
            if self_room.entryDoorPos != room_from_grid.entryDoorPos:
                return False
            if self_room.exitDoorPos != room_from_grid.exitDoorPos:
                return False

        # Agent is placed
        if self.agent_pos is None:
            return False
        if self.agent_dir is None:
            return False

        # Agent should not be on the goal
        if np.all(self.agent_pos == self.goal_pos):
            return False

        # Agent is placed in the first room
        top, size, door_positions, door_walls = self.room_info_from_encoding_and_position(self.encoding, self.agent_pos)

        if self.rooms[0].top != top:
            return False
        if self.rooms[0].size != size:
            return False
        if len(door_positions) == 2:
            return False
        if len(door_positions) == 0 and self.rooms[0].exitDoorPos is not None:
            return False
        if len(door_positions) == 1 and [self.rooms[0].exitDoorPos] != door_positions:
            return False

        # Rooms should be consistent
        if num_rooms == 1:
            if self.rooms[0].top != self.rooms[0].entryDoorPos:
                return False
            if self.rooms[0].exitDoorPos is not None:
                return False
        else:
            door_positions = []
            for idx_r in range(num_rooms):

                if idx_r == 0:
                    if self.rooms[idx_r].top != self.rooms[idx_r].entryDoorPos:
                        return False

                if 0 <= (idx_r - 1):
                    if self.rooms[idx_r - 1].exitDoorPos != self.rooms[idx_r].entryDoorPos:
                        return False

                if (idx_r + 1) < num_rooms:
                    if self.rooms[idx_r].exitDoorPos != self.rooms[idx_r + 1].entryDoorPos:
                        return False
                    door_positions.append(self.rooms[idx_r].exitDoorPos)

            door_colors = []
            # Doors exist and are closed and unlocked
            for door_pos in door_positions:
                door_obj = self.grid.get(*door_pos)
                if door_obj is None:
                    return False
                if door_obj.type != 'door':
                    return False
                if door_obj.is_open:
                    return False
                if door_obj.is_locked:
                    return False
                door_colors.append(door_obj.color)

            num_doors = len(door_colors)
            # No adjacent doors have the same colors
            if num_doors > 1:
                for idx_c in range(num_doors):
                    if 0 <= (idx_c - 1):
                        if door_colors[idx_c - 1] == door_colors[idx_c]:
                            return False

                    if (idx_c + 1) < num_doors:
                        if door_colors[idx_c] == door_colors[idx_c + 1]:
                            return False

        return True

    """ adapted from _placeRoom """
    def _sample_room_position_size_from_door_position_wall(self, doorPos, doorWall, minSz, maxSz):
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz+1)
        sizeY = self._rand_int(minSz, maxSz+1)

        # Entry on the right
        if doorWall == 0:
            topX = doorPos[0] - sizeX + 1
            y = doorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif doorWall == 1:
            x = doorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = doorPos[1] - sizeY + 1
        # Entry wall on the left
        elif doorWall == 2:
            topX = doorPos[0]
            y = doorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif doorWall == 3:
            x = doorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = doorPos[1]
        else:
            assert False, doorWall

        top = (topX, topY)
        size = (sizeX, sizeY)

        return top, size

    """ adapted from _placeRoom """
    @staticmethod
    def _is_room_in_bounds_from_position_and_size(grid_size, position, size):
        width, height = grid_size
        pos_x, pos_y = position
        size_x, size_y = size

        if pos_x < 0 or pos_y < 0:
            return False
        # Note: there is an inconsistency here between > for x and and >= for y, but keeping logic the same as _placeRoom
        if pos_x + size_x > width or pos_y + size_y >= height:
            return False

        return True

    """ adapted from _placeRoom """
    @staticmethod
    def _is_room_free_from_rooms_position_and_size(rooms, position, size):
        topX, topY = position
        sizeX, sizeY = size

        for room in rooms:
            nonOverlap = \
                topX + sizeX < room.top[0] or \
                room.top[0] + room.size[0] <= topX or \
                topY + sizeY < room.top[1] or \
                room.top[1] + room.size[1] <= topY

            if not nonOverlap:
                return False

        return True

    @staticmethod
    def _get_entry_wall(room):
        top_x, top_y = room.top
        size_x, size_y = room.size
        entry_door_x, entry_door_y = room.entryDoorPos

        if (top_x + size_x - 1) == entry_door_x:
            entry_door_wall = 0
        elif (top_y + size_y - 1) == entry_door_y:
            entry_door_wall = 1
        elif top_x == entry_door_x:
            entry_door_wall = 2
        elif top_y == entry_door_y:
            entry_door_wall = 3
        else:
            raise ValueError

        return entry_door_wall

    @staticmethod
    def _get_exit_wall(room):
        top_x, top_y = room.top
        size_x, size_y = room.size
        exit_door_x, exit_door_y = room.exitDoorPos

        if (top_x + size_x - 1) == exit_door_x:
            exit_door_wall = 0
        elif (top_y + size_y - 1) == exit_door_y:
            exit_door_wall = 1
        elif top_x == exit_door_x:
            exit_door_wall = 2
        elif top_y == exit_door_y:
            exit_door_wall = 3
        else:
            raise ValueError

        return exit_door_wall

    @staticmethod
    def _is_door_on_room_wall_from_position_and_size(door_position, door_wall, room_position, room_size):

        door_pos_x, door_pos_y = door_position
        top_x, top_y = room_position
        size_x, size_y = room_size

        # Wall on right
        if door_wall == 0:
            on_wall = top_x + size_x - 1 == door_pos_x
            inside_corners = top_y < door_pos_y and door_pos_y < (top_y + size_y - 1)
        # Wall on south
        elif door_wall == 1:
            on_wall = top_y + size_y - 1 == door_pos_y
            inside_corners = top_x < door_pos_x and door_pos_x < (top_x + size_x - 1)
        # Wall on left
        elif door_wall == 2:
            on_wall = top_x == door_pos_x
            inside_corners = top_y < door_pos_y and door_pos_y < (top_y + size_y - 1)
        # Wall on top
        elif door_wall == 3:
            on_wall = top_y == door_pos_y
            inside_corners = top_x < door_pos_x and door_pos_x < (top_x + size_x - 1)
        else:
            assert False, door_wall

        is_door_on_room_wall = on_wall and inside_corners

        return is_door_on_room_wall

    @staticmethod
    def _reverse_rooms(rooms):
        rooms_reversed = deepcopy(rooms)
        rooms_reversed.reverse()

        num_rooms = len(rooms)
        if num_rooms < 2:
            return rooms_reversed

        # last room will have None for exitDoorPos
        for idx, this_room in enumerate(rooms_reversed):
            if idx == 0:
                # first room (was the last room)
                assert this_room.exitDoorPos is None
                this_room.exitDoorPos = this_room.entryDoorPos
                this_room.entryDoorPos = this_room.top
            elif idx == (num_rooms - 1):
                # last room (was the first room)
                assert this_room.entryDoorPos == this_room.top
                this_room.entryDoorPos = this_room.exitDoorPos
                this_room.exitDoorPos = None
            else:
                # swap entry and exit door positions
                old_entry_door_pos = this_room.entryDoorPos
                old_exit_door_pos = this_room.exitDoorPos
                this_room.entryDoorPos = old_exit_door_pos
                this_room.exitDoorPos = old_entry_door_pos

        return rooms_reversed

    @staticmethod
    def _get_room_perimeter_cells(room):
        top_left_x, top_left_y = room.top
        size_x, size_y = room.size

        bottom_right_x = top_left_x + size_x - 1
        bottom_right_y = top_left_y + size_y - 1

        right_wall = [(bottom_right_x, y) for y in range(top_left_y, bottom_right_y)]
        bottom_wall = [(x, bottom_right_y) for x in range(bottom_right_x, top_left_x, -1)]
        left_wall = [(top_left_x, y) for y in range(bottom_right_y, top_left_y, -1)]
        top_wall = [(x, top_left_y) for x in range(top_left_x, bottom_right_x)]

        cells = right_wall + bottom_wall + left_wall + top_wall

        return cells

    @staticmethod
    def _get_room_membership_at_position(rooms, position):
        memberships = []
        for idx_r, r in enumerate(rooms):
            if MultiRoomEnv._is_position_in_room(r, position):
                memberships.append(idx_r)
        return memberships

    @staticmethod
    def _is_position_in_room(room, position):
        top_left_x, top_left_y = room.top
        size_x, size_y = room.size

        bottom_right_x = top_left_x + size_x - 1
        bottom_right_y = top_left_y + size_y - 1

        pos_x, pos_y = position

        is_position_in_room = (
            top_left_x <= pos_x and
            pos_x <= bottom_right_x and
            top_left_y <= pos_y and
            pos_y <= bottom_right_y
        )

        return is_position_in_room


if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname


class MultiRoomEnvN2to2S4to4(MultiRoomEnv):
    def __init__(self, seed=None):
        super().__init__(
            min_num_rooms=2,
            max_num_rooms=2,
            min_room_size=4,
            max_room_size=4,
            seed=seed,
        )


class MultiRoomEnvN3to3S4to4(MultiRoomEnv):
    def __init__(self, seed=None):
        super().__init__(
            min_num_rooms=3,
            max_num_rooms=3,
            min_room_size=4,
            max_room_size=4,
            seed=seed,
        )


class MultiRoomEnvN4to4S4to4(MultiRoomEnv):
    def __init__(self, seed=None):
        super().__init__(
            min_num_rooms=4,
            max_num_rooms=4,
            min_room_size=4,
            max_room_size=4,
            seed=seed,
        )


class MultiRoomEnvN2to2S4to4T80(MultiRoomEnv):
    def __init__(self, seed=None):
        super().__init__(
            min_num_rooms=2,
            max_num_rooms=2,
            min_room_size=4,
            max_room_size=4,
            max_steps=80,
            seed=seed,
        )


class MultiRoomEnvN3to3S4to4T80(MultiRoomEnv):
    def __init__(self, seed=None):
        super().__init__(
            min_num_rooms=3,
            max_num_rooms=3,
            min_room_size=4,
            max_room_size=4,
            max_steps=80,
            seed=seed,
        )


class MultiRoomEnvN4to4S4to4T80(MultiRoomEnv):
    def __init__(self, seed=None):
        super().__init__(
            min_num_rooms=4,
            max_num_rooms=4,
            min_room_size=4,
            max_room_size=4,
            max_steps=80,
            seed=seed,
        )


""" Re-implementation of MultiRoom-N4-Random from Prioritized Level Replay (Jiang et al ICML 2021)"""
class MultiRoomEnvN1to4S4to7G13T80(MultiRoomEnv):
    def __init__(self, seed=None):
        super().__init__(
            min_num_rooms=1,
            max_num_rooms=4,
            min_room_size=4,
            max_room_size=7,
            grid_size=13,
            max_steps=80,
            seed=seed,
        )


class MultiRoomEnvN1to1S4to7G13T80(MultiRoomEnv):
    def __init__(self, seed=None):
        super().__init__(
            min_num_rooms=1,
            max_num_rooms=1,
            min_room_size=4,
            max_room_size=7,
            grid_size=13,
            max_steps=80,
            seed=seed,
        )


class MultiRoomEnvN2to2S4to7G13T80(MultiRoomEnv):
    def __init__(self, seed=None):
        super().__init__(
            min_num_rooms=2,
            max_num_rooms=2,
            min_room_size=4,
            max_room_size=7,
            grid_size=13,
            max_steps=80,
            seed=seed,
        )


class MultiRoomEnvN3to3S4to7G13T80(MultiRoomEnv):
    def __init__(self, seed=None):
        super().__init__(
            min_num_rooms=3,
            max_num_rooms=3,
            min_room_size=4,
            max_room_size=7,
            grid_size=13,
            max_steps=80,
            seed=seed,
        )


class MultiRoomEnvN4to4S4to7G13T80(MultiRoomEnv):
    def __init__(self, seed=None):
        super().__init__(
            min_num_rooms=4,
            max_num_rooms=4,
            min_room_size=4,
            max_room_size=7,
            grid_size=13,
            max_steps=80,
            seed=seed,
        )


gym_register(
    id='MiniGrid-MultiRoom-N2to2-S4to4-v0',
    entry_point=module_path + ':MultiRoomEnvN2to2S4to4',
    max_episode_steps=MultiRoomEnvN2to2S4to4._default_max_steps(2),
)


gym_register(
    id='MiniGrid-MultiRoom-N3to3-S4to4-v0',
    entry_point=module_path + ':MultiRoomEnvN3to3S4to4',
    max_episode_steps=MultiRoomEnvN3to3S4to4._default_max_steps(3),
)


gym_register(
    id='MiniGrid-MultiRoom-N4to4-S4to4-v0',
    entry_point=module_path + ':MultiRoomEnvN4to4S4to4',
    max_episode_steps=MultiRoomEnvN4to4S4to4._default_max_steps(4),
)


gym_register(
    id='MiniGrid-MultiRoom-N2to2-S4to4-T80-v0',
    entry_point=module_path + ':MultiRoomEnvN2to2S4to4T80',
    max_episode_steps=80,
)


gym_register(
    id='MiniGrid-MultiRoom-N3to3-S4to4-T80-v0',
    entry_point=module_path + ':MultiRoomEnvN3to3S4to4T80',
    max_episode_steps=80,
)


gym_register(
    id='MiniGrid-MultiRoom-N4to4-S4to4-T80-v0',
    entry_point=module_path + ':MultiRoomEnvN4to4S4to4T80',
    max_episode_steps=80,
)


gym_register(
    id='MiniGrid-MultiRoom-N1to4-S4to7-G13-T80-v0',
    entry_point=module_path + ':MultiRoomEnvN1to4S4to7G13T80',
    max_episode_steps=80,
)


gym_register(
    id='MiniGrid-MultiRoom-N1to1-S4to7-G13-T80-v0',
    entry_point=module_path + ':MultiRoomEnvN1to1S4to7G13T80',
    max_episode_steps=80,
)


gym_register(
    id='MiniGrid-MultiRoom-N2to2-S4to7-G13-T80-v0',
    entry_point=module_path + ':MultiRoomEnvN2to2S4to7G13T80',
    max_episode_steps=80,
)


gym_register(
    id='MiniGrid-MultiRoom-N3to3-S4to7-G13-T80-v0',
    entry_point=module_path + ':MultiRoomEnvN3to3S4to7G13T80',
    max_episode_steps=80,
)


gym_register(
    id='MiniGrid-MultiRoom-N4to4-S4to7-G13-T80-v0',
    entry_point=module_path + ':MultiRoomEnvN4to4S4to7G13T80',
    max_episode_steps=80,
)
