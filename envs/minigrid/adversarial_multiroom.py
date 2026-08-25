from copy import deepcopy
import numpy as np

from gym import spaces
from gym_minigrid.minigrid import COLOR_NAMES, OBJECT_TO_IDX, Door, Goal, Grid, Wall

from envs.registration import register as gym_register

from .multiroom import MultiRoomEnv


class AdversarialMultiRoomEnv(MultiRoomEnv):
    def __init__(
            self,
            min_num_rooms,
            max_num_rooms,
            min_room_size=4,
            max_room_size=10,
            grid_size=25,
            min_room_mode=False,
            max_steps=None,
            seed=None,
            random_z_dim=50,
        ):

        self.random_z_dim = random_z_dim
        self.min_room_mode = min_room_mode
        self.adversary_max_steps = 1 if self.min_room_mode else max_num_rooms - min_num_rooms
        self.adversary_action_space = spaces.Discrete(2)

        super().__init__(min_num_rooms, max_num_rooms, min_room_size, max_room_size, grid_size, max_steps, seed)

        # from multigrid adversarial
        self.adversary_ts_obs_space = spaces.Box(
            low=0, high=self.adversary_max_steps, shape=(1,), dtype='uint8',
        )
        self.adversary_randomz_obs_space = spaces.Box(
            low=0, high=1.0, shape=(self.random_z_dim,), dtype=np.float32,
        )
        self.adversary_image_obs_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.width, self.height, 3),
            dtype='uint8',
        )
        self.adversary_observation_space = spaces.Dict({
            'image': self.adversary_image_obs_space,
            'time_step': self.adversary_ts_obs_space,
            'random_z': self.adversary_randomz_obs_space
        })

    @property
    def processed_action_dim(self):
        # If I understand this correctly, this just means we will expand actions depending on the action space
        return 1

    """ based on multigrid adversarial """
    def _generate_random_z(self):
        random_z = np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)
        return random_z

    def sample_adversary_action_space(self):
        return self.adversary_action_space.sample()

    def reset(self, reset_to_empty=False):
        self.reset_agent_status()
        self.goal_pos = None

        self.grid = Grid(self.width, self.height)

        if not reset_to_empty:
            # Place minimum number of rooms to prepare for adversary steps
            self._set_rooms(self.width, self.height, num_rooms=self.min_num_rooms)

        self._set_mission()

        self.agent_start_pos = None
        self.agent_start_dir = None
        self.step_count = 0
        self.adversary_step_count = 0

        # self.done = False

        adv_obs = self.gen_adversary_obs()

        return adv_obs

    def reset_agent_status(self):
        self.agent_pos = None
        self.agent_dir = None
        self.carrying = None

    def reset_agent(self):
        return self.restart_level()

    """ restarts the level to the beginning, resetting agent and doors """
    def restart_level(self):
        assert self.agent_start_pos is not None
        assert self.agent_start_dir is not None
        assert len(self.rooms) > 0

        self.reset_agent_status()

        # Set agent
        self.agent_dir = self.agent_start_dir
        self.place_agent(top=self.agent_start_pos, size=(1, 1), rand_dir=False)

        # Reset doors
        for room_idx in range(len(self.rooms) - 1):
            exit_door_pos = self.rooms[room_idx].exitDoorPos
            self.grid.get(*exit_door_pos).is_open = False

        self.step_count = 0

        obs = self.gen_obs()

        return obs

    def reset_to_level(self, level):
        obs = self.reset_to_encoding(level)
        return obs

    """ reset to level encoding. assumes the encoded level is from the start of the episode """
    def reset_to_encoding(self, encoded_level):
        self.reset(reset_to_empty=True)

        width, height, channels = encoded_level.shape
        assert width == self.width, \
            f"Expected encoded level width ({width}) to be the same as the environment width ({self.width}), but it is not."
        assert height == self.height, \
            f"Expected encoded level height ({height}) to be the same as the environment height ({self.height}), but it is not."
        assert channels == 3, \
            f"Expected encoded level to have 3 channels, but it has {channels}."

        # Currently, we require the agent and the goal to be specified to determine the ordering of the corridor
        found_agent = False
        found_agent_pos = None
        found_agent_dir = None
        found_agent_on_door = False

        found_goal = False
        found_goal_pos = None

        for i in range(width):
            for j in range(height):
                if encoded_level[i, j, 0] == OBJECT_TO_IDX['agent']:
                    assert encoded_level[i, j, 1] == 0
                    assert not found_agent

                    found_agent = True
                    found_agent_pos = (i,j)
                    found_agent_dir = encoded_level[i, j, 2]
                elif encoded_level[i, j, 0] == OBJECT_TO_IDX['door']:
                    # check if the agent is on an open door
                    if encoded_level[i, j, 2] >= 10:
                        assert not found_agent

                        found_agent = True
                        found_agent_pos = (i,j)
                        assert encoded_level[i, j, 2] % 10 == 0
                        found_agent_dir = encoded_level[i, j, 2] // 10
                        found_agent_on_door = True
                elif encoded_level[i, j, 0] == OBJECT_TO_IDX['goal']:
                    assert not found_goal

                    found_goal = True
                    found_goal_pos = (i,j)

        assert found_agent, \
            f"Resetting to encoding requires that the encoding contains one agent, but an agent was not found."
        assert found_goal, \
            f"Resetting to encoding requires that the encoding contains one goal, but a goal was not found."

        # Remove agent from the encoding
        encoded_level_wo_agent = encoded_level
        if not found_agent_on_door:
            encoded_level_wo_agent[found_agent_pos[0], found_agent_pos[1], 0] = OBJECT_TO_IDX['empty']
            encoded_level_wo_agent[found_agent_pos[0], found_agent_pos[1], 1] = 0
            encoded_level_wo_agent[found_agent_pos[0], found_agent_pos[1], 2] = 0
        else:
            encoded_level_wo_agent[found_agent_pos[0], found_agent_pos[1], 2] = 0

        # Set rooms
        room_list = self.rooms_from_encoding_and_position_in_last_room(encoded_level_wo_agent, found_goal_pos)
        self.rooms = room_list

        # Set grid from encoding
        new_grid, _ = self.grid.decode(encoded_level_wo_agent)
        self.grid = new_grid

        # Set agent - note this assumes that the encoding is of the initial scene
        self.agent_pos = found_agent_pos
        self.agent_dir = found_agent_dir
        self.agent_start_pos = self.agent_pos
        self.agent_start_dir = self.agent_dir

        # Set goal
        self.goal_pos = found_goal_pos

        return self.reset_agent()

    def mutate_level(self, num_edits=1):

        assert self.step_count == 0, \
            f"Mutating a level is only possible before the agent has acted."
        assert len(self.rooms) > 0, \
            f"Mutating a level is only possible when rooms have initially been placed."

        for _ in range(num_edits):

            num_rooms = len(self.rooms)

            if num_rooms < self.max_num_rooms:
                # can add another room
                idx_max_room = num_rooms
            elif num_rooms == self.max_num_rooms:
                # can't add another room
                idx_max_room = num_rooms - 1
            else:
                assert False

            if self.min_num_rooms < num_rooms:
                # can remove a room
                idx_min_room = -1
            elif self.min_num_rooms == num_rooms:
                # can't remove a room
                idx_min_room = 0
            else:
                assert False

            # select room to mutate
            idx_room_edit = self._rand_int(idx_min_room, idx_max_room + 1)

            if idx_room_edit == num_rooms:
                # add a room
                add_room_to_end = self._rand_bool()

                # Temporarily remove the goal
                self.grid.set(*self.goal_pos, None)
                goal_pos_before_edit = self.goal_pos
                self.goal_pos = None

                if add_room_to_end:
                    # add room to the end
                    is_room_added = self._add_room()

                    # randomly set new goal position
                    self._set_goal_position()
                else:
                    # add room to the front
                    is_room_added = self._add_room_to_front()

                    num_placement_tries = 1000
                    is_placement_successful = False

                    for _ in range(num_placement_tries):
                        # place agent in new first room, but not where the goal was
                        self._set_agent()
                        is_placement_successful = not np.all(self.agent_pos == goal_pos_before_edit)
                        if is_placement_successful:
                            break
                    assert is_placement_successful

                    self.agent_start_pos = self.agent_pos
                    self.agent_start_dir = self.agent_dir

                    # Replace goal at the same place
                    assert self.grid.get(*goal_pos_before_edit) is None
                    self.goal_pos = self.place_obj(Goal(), goal_pos_before_edit, (1, 1))

            elif idx_room_edit == -1:
                # remove a room
                remove_room_from_end = self._rand_bool()

                # Temporarily remove the goal
                self.grid.set(*self.goal_pos, None)
                goal_pos_before_edit = self.goal_pos
                self.goal_pos = None

                if remove_room_from_end:
                    # remove room at the end
                    self._remove_room()

                    # randomly set new goal position
                    self._set_goal_position()
                else:
                    # remove room at the front
                    self._remove_room_from_front()

                    num_placement_tries = 1000
                    is_placement_successful = False

                    for _ in range(num_placement_tries):
                        # place agent in new first room, but not where the goal was
                        self._set_agent()
                        is_placement_successful = not np.all(self.agent_pos == goal_pos_before_edit)
                        if is_placement_successful:
                            break
                    assert is_placement_successful

                    self.agent_start_pos = self.agent_pos
                    self.agent_start_dir = self.agent_dir

                    # Replace goal at the same place
                    assert self.grid.get(*goal_pos_before_edit) is None
                    self.goal_pos = self.place_obj(Goal(), goal_pos_before_edit, (1, 1))

            else:
                # respawn an existing room

                this_room = self.rooms[idx_room_edit]
                top_x_before_edit, top_y_before_edit = this_room.top
                size_x_before_edit, size_y_before_edit = this_room.size

                perimeter_cells = self._get_room_perimeter_cells(this_room)

                respawn_agent = idx_room_edit == 0
                respawn_goal = idx_room_edit == num_rooms - 1

                if respawn_goal:
                    # Temporarily remove the goal
                    self.grid.set(*self.goal_pos, None)
                    goal_pos_before_edit = self.goal_pos
                    self.goal_pos = None

                if num_rooms == 1:
                    # respawn the first room
                    assert idx_room_edit == 0

                    num_placement_tries = 1000
                    is_placement_successful = False

                    for _ in range(num_placement_tries):

                        cell_range = int(np.ceil((self.grid_size * 0.1) / 0.5))
                        if cell_range % 2 == 0:
                            cell_range_half = cell_range // 2

                            new_top_x_delta = self._rand_int(-cell_range_half, cell_range_half + 1)
                            new_top_y_delta = self._rand_int(-cell_range_half, cell_range_half + 1)

                        else:
                            cell_range_half = (cell_range - 1) // 2

                            coin_flip = self._rand_bool()
                            not_coin_flip = not coin_flip

                            new_top_x_delta = self._rand_int(-cell_range_half - int(not_coin_flip), cell_range_half + int(coin_flip) + 1)

                            coin_flip = self._rand_bool()
                            not_coin_flip = not coin_flip

                            new_top_y_delta = self._rand_int(-cell_range_half - int(not_coin_flip), cell_range_half + int(coin_flip) + 1)

                        new_top_x  = top_x_before_edit + new_top_x_delta
                        new_top_y = top_y_before_edit + new_top_y_delta
                        new_top = (new_top_x, new_top_y)

                        new_size_x = self._rand_int(self.min_room_size, self.max_room_size + 1)
                        new_size_y = self._rand_int(self.min_room_size, self.max_room_size + 1)
                        new_size = (new_size_x, new_size_y)

                        is_placement_successful = self._is_room_in_bounds_from_position_and_size(
                            (self.grid_size, self.grid_size),
                            new_top,
                            new_size,
                        )

                        if is_placement_successful:
                            break

                    if not is_placement_successful:
                        # Couldn't place, use previous room
                        new_top = (top_x_before_edit, top_y_before_edit)
                        new_size = (size_x_before_edit, size_y_before_edit)

                    # remove wall
                    for c in perimeter_cells:
                        self.grid.set(*c, None)

                    # modify room
                    this_room.top = new_top
                    this_room.size = new_size
                    this_room.entryDoorPos = new_top
                    assert this_room.exitDoorPos is None

                    # add wall
                    new_perimeter_cells = self._get_room_perimeter_cells(this_room)
                    wall = Wall()
                    for c in new_perimeter_cells:
                        self.grid.set(*c, wall)

                else:
                    # respawn a room when other rooms are already present

                    num_placement_tries = 1000
                    is_placement_successful = False

                    for _ in range(num_placement_tries):

                        if idx_room_edit == 0:
                            # initial room - work backwards from exit
                            exit_wall = self._get_exit_wall(this_room)
                            new_top, new_size = self._sample_room_position_size_from_door_position_wall(this_room.exitDoorPos, exit_wall, self.min_room_size, self.max_room_size)

                            check_rooms = self.rooms[2:]

                            recolor_entry_door = False
                            entry_door_color_to_avoid = None
                            recolor_exit_door = True
                            if num_rooms < 3:
                                exit_door_color_to_avoid = None
                            else:
                                exit_door_color_to_avoid = self.grid.get(*self.rooms[1].exitDoorPos).color

                        elif idx_room_edit == num_rooms - 1:
                            # last room
                            entry_wall = self._get_entry_wall(this_room)
                            new_top, new_size = self._sample_room_position_size_from_door_position_wall(this_room.entryDoorPos, entry_wall, self.min_room_size, self.max_room_size)

                            check_rooms = self.rooms[:-2]

                            recolor_entry_door = True
                            if num_rooms < 3:
                                entry_door_color_to_avoid = None
                            else:
                                entry_door_color_to_avoid = self.grid.get(*self.rooms[-3].exitDoorPos).color
                            recolor_exit_door = False
                            exit_door_color_to_avoid = None

                        else:
                            # middle room - there will be two doors
                            entry_wall = self._get_entry_wall(this_room)
                            exit_wall = self._get_exit_wall(this_room)

                            num_sub_placement_tries = 1000
                            is_sub_placement_successful = False

                            for _ in range(num_sub_placement_tries):

                                new_top, new_size = self._sample_room_position_size_from_door_position_wall(this_room.entryDoorPos, entry_wall, self.min_room_size, self.max_room_size)

                                is_sub_placement_successful = self._is_door_on_room_wall_from_position_and_size(
                                    this_room.exitDoorPos,
                                    exit_wall,
                                    new_top,
                                    new_size,
                                )

                                if is_sub_placement_successful:
                                    break

                            if not is_sub_placement_successful:
                                # Couldn't place, use previous room
                                new_top = (top_x_before_edit, top_y_before_edit)
                                new_size = (size_x_before_edit, size_y_before_edit)

                            check_rooms = self.rooms[:(idx_room_edit - 1)] + self.rooms[(idx_room_edit + 2):]

                            recolor_entry_door = True
                            if 0 <= idx_room_edit - 2:
                                entry_door_color_to_avoid = self.grid.get(*self.rooms[idx_room_edit - 2].exitDoorPos).color
                            else:
                                entry_door_color_to_avoid = None
                            recolor_exit_door = True
                            if idx_room_edit + 2 < num_rooms:
                                exit_door_color_to_avoid = self.grid.get(*self.rooms[idx_room_edit + 2].entryDoorPos).color
                            else:
                                exit_door_color_to_avoid = None

                        is_room_in_bounds = self._is_room_in_bounds_from_position_and_size(
                            (self.grid_size, self.grid_size),
                            new_top,
                            new_size,
                        )

                        is_room_free = self._is_room_free_from_rooms_position_and_size(
                            check_rooms,
                            new_top,
                            new_size,
                        )

                        is_placement_successful = is_room_in_bounds and is_room_free

                        if is_placement_successful:
                            break

                    if not is_placement_successful:
                        # Couldn't place, use previous room
                        new_top = (top_x_before_edit, top_y_before_edit)
                        new_size = (size_x_before_edit, size_y_before_edit)

                    #remove wall
                    for c in perimeter_cells:
                        assert self.grid.get(*c).type in ['wall', 'door']

                        room_membership = self._get_room_membership_at_position(self.rooms, c)
                        if room_membership == [idx_room_edit]:
                            # sole membership to the room to be removed, remove wall
                            self.grid.set(*c, None)

                    # modify room
                    this_room.top = new_top
                    this_room.size = new_size
                    if idx_room_edit == 0:
                        # also need to update the entry door
                        this_room.entryDoorPos = new_top

                    # add wall
                    new_perimeter_cells = self._get_room_perimeter_cells(this_room)
                    wall = Wall()
                    for c in new_perimeter_cells:
                        obj = self.grid.get(*c)
                        if obj is None or obj.type != 'door':
                            self.grid.set(*c, wall)

                    # recolor the doors
                    if recolor_entry_door:
                        door_colors = set(COLOR_NAMES)
                        if entry_door_color_to_avoid is not None:
                            door_colors.remove(entry_door_color_to_avoid)

                        # Note: the use of sorting here guarantees determinism,
                        # This is needed because Python's set is not deterministic
                        new_entry_door_color = self._rand_elem(sorted(door_colors))

                        new_entry_door = Door(new_entry_door_color)
                        self.grid.set(*this_room.entryDoorPos, new_entry_door)

                    if recolor_exit_door:
                        door_colors = set(COLOR_NAMES)
                        if exit_door_color_to_avoid is not None:
                            door_colors.remove(exit_door_color_to_avoid)
                        # remove new entry door color if necessary
                        if recolor_entry_door and new_entry_door_color in door_colors:
                            door_colors.remove(new_entry_door_color)

                        # Note: the use of sorting here guarantees determinism,
                        # This is needed because Python's set is not deterministic
                        new_exit_door_color = self._rand_elem(sorted(door_colors))

                        new_exit_door = Door(new_exit_door_color)
                        self.grid.set(*this_room.exitDoorPos, new_exit_door)

                if respawn_agent:
                    self._set_agent()
                    self.agent_start_pos = self.agent_pos
                    self.agent_start_dir = self.agent_dir

                if respawn_goal:
                    self._set_goal_position()

        return self.reset_agent()

    def gen_adversary_obs(self):
        grid_image = self.encoding
        adv_obs = {
            'image': grid_image,
            'time_step': [self.adversary_step_count],
            'random_z': self._generate_random_z()
        }
        return adv_obs

    """ Incrementally build the multiroom maze. """
    def step_adversary(self, action):
        if (not self.min_room_mode) and (self.adversary_step_count < self.adversary_max_steps):
            add_room = action == 1
            if add_room:
                is_placement_successful = self._add_room()

        self.adversary_step_count += 1

        if self.adversary_step_count == self.adversary_max_steps:
            # in the final step of the adversary, we place the agent and the goal
            self._set_agent()
            self._set_goal_position()
            self.agent_start_pos = self.agent_pos
            self.agent_start_dir = self.agent_dir

        adv_obs = self.gen_adversary_obs()
        adv_done = self.adversary_step_count >= self.adversary_max_steps
        adv_reward = 0.
        adv_info = {}

        return adv_obs, adv_reward, adv_done, adv_info

    def _add_room(self, prevent_right_exit=False):
        assert self.goal_pos is None

        prev_room = self.rooms[-1]
        assert prev_room.exitDoorPos is None

        num_rooms_left = 1

        entryDoorWall = self._get_entry_wall(prev_room)
        entry_door_pos = prev_room.entryDoorPos

        topX, topY = prev_room.top
        sizeX, sizeY = prev_room.size

        num_placement_tries = 1000

        for ctr_this_try in range(num_placement_tries):

            # from _placeRoom

            # Pick which wall to place the out door on
            wallSet = set((0, 1, 2, 3))
            wallSet.remove(entryDoorWall)
            if prevent_right_exit:
                if 0 in wallSet:
                    wallSet.remove(0)
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

            is_placement_successful = self._placeRoom(
                num_rooms_left,
                self.rooms,
                self.min_room_size,
                self.max_room_size,
                nextEntryWall,
                exitDoorPos,
            )

            if is_placement_successful:
                break

        # exit if we can't place any more rooms
        if not is_placement_successful:
            return False

        # Update grid
        wall = Wall()

        new_room = self.rooms[-1]
        topX, topY = new_room.top
        sizeX, sizeY = new_room.size

        # Draw the top and bottom walls
        for i in range(0, sizeX):
            self.grid.set(topX + i, topY, wall)
            self.grid.set(topX + i, topY + sizeY - 1, wall)

        # Draw the left and right walls
        for j in range(0, sizeY):
            self.grid.set(topX, topY + j, wall)
            self.grid.set(topX + sizeX - 1, topY + j, wall)

        prevDoorColor = self.grid.get(*entry_door_pos).color
        doorColors = set(COLOR_NAMES)
        doorColors.remove(prevDoorColor)

        # Note: the use of sorting here guarantees determinism,
        # This is needed because Python's set is not deterministic
        doorColor = self._rand_elem(sorted(doorColors))

        entryDoor = Door(doorColor)
        self.grid.set(*new_room.entryDoorPos, entryDoor)

        prev_room.exitDoorPos = new_room.entryDoorPos

        return True

    def _add_room_to_front(self):

        # reverse rooms
        self.rooms = self._reverse_rooms(self.rooms)

        # add room at the end of the reversed rooms (this modifies self.rooms)
        # we need to prevent the case where the new entry room exits left,
        # which means we prevent right exit on the new end room in the reversed rooms
        is_placement_successful = self._add_room(prevent_right_exit=True)

        # reverse rooms again
        self.rooms = self._reverse_rooms(self.rooms)

        return is_placement_successful

    def _remove_room(self):
        assert self.goal_pos is None
        assert len(self.rooms) > 1

        idx_last_room = len(self.rooms) - 1

        last_room_perimeter_cells = self._get_room_perimeter_cells(self.rooms[idx_last_room])
        for c in last_room_perimeter_cells:
            assert self.grid.get(*c).type in ['wall', 'door']

            room_membership = self._get_room_membership_at_position(self.rooms, c)
            if room_membership == [idx_last_room]:
                # sole membership to the room to be removed, remove wall
                self.grid.set(*c, None)

        # remove entry door
        wall = Wall()
        self.grid.set(*self.rooms[idx_last_room].entryDoorPos, wall)
        self.rooms[idx_last_room - 1].exitDoorPos = None

        self.rooms = self.rooms[:-1]

    def _remove_room_from_front(self):
        assert self.goal_pos is None
        assert len(self.rooms) > 1

        # reverse rooms
        self.rooms = self._reverse_rooms(self.rooms)

        # remove the room at the front (reversed end)
        self._remove_room()

        # reverse rooms
        self.rooms = self._reverse_rooms(self.rooms)

    def reset_random(self, num_rooms=None):
        self.reset(reset_to_empty=True)

        self._set_rooms(self.width, self.height, num_rooms=num_rooms)

        self._set_agent()

        self._set_goal_position()

        self.agent_start_pos = self.agent_pos
        self.agent_start_dir = self.agent_dir

        return self.reset_agent()

    def reset_to_params(self, params):
        assert np.isscalar(params)
        return self.reset_random(num_rooms=params)


class MultiRoomEnvN2to4S4to4Adversarial(AdversarialMultiRoomEnv):
    def __init__(self, seed=None):
        super().__init__(
            min_num_rooms=2,
            max_num_rooms=4,
            min_room_size=4,
            max_room_size=4,
            seed=seed,
        )


class MultiRoomEnvN2to10S4to4Adversarial(AdversarialMultiRoomEnv):
    def __init__(self, seed=None):
        super().__init__(
            min_num_rooms=2,
            max_num_rooms=10,
            min_room_size=4,
            max_room_size=4,
            seed=seed,
        )


""" Adversarial re-implementation of MultiRoom-N4-Random from Prioritized Level Replay (Jiang et al ICML 2021)"""
class MultiRoomEnvN1to4S4to7G13T80Adversarial(AdversarialMultiRoomEnv):
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


class MultiRoomEnvN1to4S4to7G13T80MinRoomAdversarial(AdversarialMultiRoomEnv):
    def __init__(self, seed=None):
        super().__init__(
            min_num_rooms=1,
            max_num_rooms=4,
            min_room_size=4,
            max_room_size=7,
            grid_size=13,
            max_steps=80,
            min_room_mode=True,
            seed=seed,
        )


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname


gym_register(
    id='MiniGrid-MultiRoom-N2to4-S4to4-Adversarial-v0',
    entry_point=module_path + ':MultiRoomEnvN2to4S4to4Adversarial',
    max_episode_steps=MultiRoomEnvN2to4S4to4Adversarial._default_max_steps(4),
)


gym_register(
    id='MiniGrid-MultiRoom-N2to10-S4to4-Adversarial-v0',
    entry_point=module_path + ':MultiRoomEnvN2to10S4to4Adversarial',
    max_episode_steps=MultiRoomEnvN2to10S4to4Adversarial._default_max_steps(10),
)


gym_register(
    id='MiniGrid-MultiRoom-N1to4-S4to7-G13-T80-Adversarial-v0',
    entry_point=module_path + ':MultiRoomEnvN1to4S4to7G13T80Adversarial',
    max_episode_steps=80,
)


gym_register(
    id='MiniGrid-MultiRoom-N1to4-S4to7-G13-T80-MinRoom-Adversarial-v0',
    entry_point=module_path + ':MultiRoomEnvN1to4S4to7G13T80MinRoomAdversarial',
    max_episode_steps=80,
)
