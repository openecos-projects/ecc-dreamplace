# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

##
# @file   Params.py
# @author Yibo Lin
# @date   Apr 2018
# @brief  User parameters
#

import os
import sys
import json
import math
import logging
from collections import OrderedDict
import pdb


class Params:
    """
    @brief Parameter class
    """

    def __init__(self):
        """
        @brief initialization
        """
        pass

    def printWelcome(self):
        """
        @brief print welcome message
        """
        content = """
========================================================
                       DREAMPlace
            Yibo Lin (http://yibolin.com)
   David Z. Pan (http://users.ece.utexas.edu/~dpan)
========================================================
"""
        logging.info(content)

    def printHelp(self):
        """
        @brief print help message for JSON parameters
        """
        content = self.toMarkdownTable()
        logging.info(content)

    def toMarkdownTable(self):
        """
        @brief convert to markdown table
        """
        key_length = len("JSON Parameter")
        key_length_map = []
        default_length = len("Default")
        default_length_map = []
        description_length = len("Description")
        description_length_map = []

        def getDefaultColumn(key, value):
            if sys.version_info.major < 3:  # python 2
                flag = isinstance(value["default"], unicode)
            else:  # python 3
                flag = isinstance(value["default"], str)
            if flag and not value["default"] and "required" in value:
                return value["required"]
            else:
                return value["default"]

        for key, value in self.params_dict.items():
            key_length_map.append(len(key))
            default_length_map.append(len(str(getDefaultColumn(key, value))))
            description_length_map.append(len(value["description"]))
            key_length = max(key_length, key_length_map[-1])
            default_length = max(default_length, default_length_map[-1])
            description_length = max(description_length, description_length_map[-1])

        content = "| %s %s| %s %s| %s %s|\n" % (
            "JSON Parameter",
            " " * (key_length - len("JSON Parameter") + 1),
            "Default",
            " " * (default_length - len("Default") + 1),
            "Description",
            " " * (description_length - len("Description") + 1),
        )
        content += "| %s | %s | %s |\n" % (
            "-" * (key_length + 1),
            "-" * (default_length + 1),
            "-" * (description_length + 1),
        )
        count = 0
        for key, value in self.params_dict.items():
            content += "| %s %s| %s %s| %s %s|\n" % (
                key,
                " " * (key_length - key_length_map[count] + 1),
                str(getDefaultColumn(key, value)),
                " " * (default_length - default_length_map[count] + 1),
                value["description"],
                " " * (description_length - description_length_map[count] + 1),
            )
            count += 1
        return content

    def toJson(self):
        """
        @brief convert to json
        """
        data = {}
        for key, value in self.__dict__.items():
            if key != "params_dict":
                data[key] = value
        return data

    @staticmethod
    def _is_enabled(value):
        if isinstance(value, str):
            return value.strip().lower() not in ("", "0", "false", "no", "off")
        return bool(value)

    def apply_l_shape_routability_preset(self):
        """
        Fill conservative hard-GGR L-shape defaults only when L-shape routability
        is enabled. Explicit user values are preserved.
        """
        if not self._is_enabled(getattr(self, "l_shape_routability_flag", False)):
            return

        defaults = {
            "l_direction_use_gpugr": 1,
            "l_shape_use_ggr_topology": 1,
            "l_shape_capacity_al_enable": 1,
            "soft_l_assignment": 0,
            "l_shape_grad_target_ratio": 0.1,
            "l_shape_grad_target_ratio_max": 0.1,
            "l_shape_overflow_threshold": 0.3,
            "l_shape_keep_during_inflation": 1,
            "l_shape_plot_flag": 0,
        }
        for key, value in defaults.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def normalize_iopin_density_weight(self):
        value = getattr(self, "iopin_density_weight", 3.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("iopin_density_weight must be numeric")
        if not math.isfinite(value):
            raise ValueError("iopin_density_weight must be finite")
        if value < 0:
            raise ValueError("iopin_density_weight must be non-negative")
        self.iopin_density_weight = value

    def normalize_m2_pg_rail_density_weight(self):
        value = getattr(self, "m2_pg_rail_density_weight", 1.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("m2_pg_rail_density_weight must be numeric")
        if not math.isfinite(value):
            raise ValueError("m2_pg_rail_density_weight must be finite")
        if value < 0:
            raise ValueError("m2_pg_rail_density_weight must be non-negative")
        self.m2_pg_rail_density_weight = value

    def normalize_removed_l_shape_weight_schedule(self):
        legacy_keys = (
            "l_shape_use_xplace_weight_schedule",
            "l_shape_num_route_iter",
            "l_shape_weight_schedule_r",
            "l_shape_weight_schedule_half_iter",
        )
        legacy_enable = getattr(self, legacy_keys[0], 0)
        if self._is_enabled(legacy_enable):
            raise ValueError(
                "l_shape_use_xplace_weight_schedule has been removed; "
                "the adaptive target-ratio controller is always used"
            )
        for key in legacy_keys:
            self.__dict__.pop(key, None)

    def normalize_m2_pg_rail_legalization_displacement_weight(self):
        value = getattr(
            self, "m2_pg_rail_legalization_displacement_weight", 0.01
        )
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                "m2_pg_rail_legalization_displacement_weight must be numeric"
            )
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                "m2_pg_rail_legalization_displacement_weight must be finite "
                "and non-negative"
            )
        self.m2_pg_rail_legalization_displacement_weight = value

    def normalize_m2_pg_rail_legalization_mode(self):
        value = getattr(self, "m2_pg_rail_legalization_mode", "soft")
        if not isinstance(value, str):
            raise ValueError(
                "m2_pg_rail_legalization_mode must be 'soft', 'hybrid_hard', "
                "or 'subset_hard'"
            )
        value = value.strip().lower()
        if value not in ("soft", "hybrid_hard", "subset_hard"):
            raise ValueError(
                "m2_pg_rail_legalization_mode must be 'soft', 'hybrid_hard', "
                "or 'subset_hard'"
            )
        self.m2_pg_rail_legalization_mode = value

    def normalize_m2_pg_rail_legalization_rail_range(self):
        def normalize_index(name, default, allow_zero=False):
            raw_value = getattr(self, name, default)
            if isinstance(raw_value, bool):
                raise ValueError("%s must be an integer" % name)
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                raise ValueError("%s must be an integer" % name)
            minimum = 0 if allow_zero else 1
            if (
                not math.isfinite(numeric_value)
                or not numeric_value.is_integer()
                or numeric_value < minimum
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError("%s must be a %s integer" % (name, qualifier))
            return int(numeric_value)

        start = normalize_index(
            "m2_pg_rail_legalization_hard_rail_start", 1
        )
        end = normalize_index(
            "m2_pg_rail_legalization_hard_rail_end", 0, allow_zero=True
        )
        if self.m2_pg_rail_legalization_mode == "subset_hard" and end < start:
            raise ValueError(
                "m2_pg_rail_legalization_hard_rail_end must be greater than "
                "or equal to m2_pg_rail_legalization_hard_rail_start in "
                "subset_hard mode"
            )
        self.m2_pg_rail_legalization_hard_rail_start = start
        self.m2_pg_rail_legalization_hard_rail_end = end

    def normalize_m2_pa_refine_limits(self):
        raw_neighbors = getattr(self, "m2_pa_refine_max_neighbors", 5)
        if isinstance(raw_neighbors, bool):
            raise ValueError("m2_pa_refine_max_neighbors must be a positive integer")
        try:
            numeric_neighbors = float(raw_neighbors)
        except (TypeError, ValueError):
            raise ValueError("m2_pa_refine_max_neighbors must be a positive integer")
        if (
            not math.isfinite(numeric_neighbors)
            or not numeric_neighbors.is_integer()
            or numeric_neighbors <= 0
        ):
            raise ValueError("m2_pa_refine_max_neighbors must be a positive integer")
        neighbors = int(numeric_neighbors)
        self.m2_pa_refine_max_neighbors = neighbors

        raw_displacement = getattr(
            self, "m2_pa_refine_max_displacement_sites", 50
        )
        try:
            displacement = float(raw_displacement)
        except (TypeError, ValueError):
            raise ValueError(
                "m2_pa_refine_max_displacement_sites must be numeric"
            )
        if not math.isfinite(displacement) or displacement < 0:
            raise ValueError(
                "m2_pa_refine_max_displacement_sites must be finite and non-negative"
            )
        self.m2_pa_refine_max_displacement_sites = displacement

    def normalize_post_legalization_adaptive_padding(self):
        for name, default in (
            ("post_legalization_adaptive_padding_flag", 0),
            ("post_legalization_padding_skip_m1_route", 1),
            ("post_legalization_padding_save_artifacts", 0),
        ):
            setattr(
                self,
                name,
                int(self._is_enabled(getattr(self, name, default))),
            )

        for name, default in (
            ("post_legalization_padding_hot_cell_ratio", 0.2),
            ("post_legalization_padding_row_free_ratio", 0.5),
        ):
            try:
                value = float(getattr(self, name, default))
            except (TypeError, ValueError):
                raise ValueError("%s must be numeric" % name)
            if not math.isfinite(value) or value < 0 or value > 1:
                raise ValueError("%s must be finite and in [0, 1]" % name)
            setattr(self, name, value)

        for name, default in (
            ("post_legalization_padding_max_sites", 1),
            ("post_legalization_padding_max_retries", 4),
            ("post_legalization_padding_rrr_iters", 0),
        ):
            raw_value = getattr(self, name, default)
            if isinstance(raw_value, bool):
                raise ValueError("%s must be a non-negative integer" % name)
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                raise ValueError("%s must be a non-negative integer" % name)
            if (
                not math.isfinite(numeric_value)
                or not numeric_value.is_integer()
                or numeric_value < 0
            ):
                raise ValueError("%s must be a non-negative integer" % name)
            setattr(self, name, int(numeric_value))

        smooth_kernel = getattr(
            self, "post_legalization_padding_smooth_kernel", 3
        )
        if isinstance(smooth_kernel, bool):
            raise ValueError(
                "post_legalization_padding_smooth_kernel must be a positive odd integer"
            )
        try:
            numeric_kernel = float(smooth_kernel)
        except (TypeError, ValueError):
            raise ValueError(
                "post_legalization_padding_smooth_kernel must be a positive odd integer"
            )
        if (
            not math.isfinite(numeric_kernel)
            or not numeric_kernel.is_integer()
            or int(numeric_kernel) <= 0
            or int(numeric_kernel) % 2 == 0
        ):
            raise ValueError(
                "post_legalization_padding_smooth_kernel must be a positive odd integer"
            )
        self.post_legalization_padding_smooth_kernel = int(numeric_kernel)

    def normalize_legalize_before_each_inflation(self):
        self.legalize_before_each_inflation_flag = int(
            self._is_enabled(
                getattr(self, "legalize_before_each_inflation_flag", 0)
            )
        )

    def normalize_inflation_area_budget_ratio(self):
        name = "inflation_area_budget_ratio"
        try:
            value = float(getattr(self, name, 0.1))
        except (TypeError, ValueError):
            raise ValueError("%s must be numeric" % name)
        if not math.isfinite(value) or value < 0:
            raise ValueError("%s must be finite and non-negative" % name)
        self.inflation_area_budget_ratio = value

    def normalize_gpugr_area_adjust_congestion_mode(self):
        name = "gpugr_area_adjust_congestion_mode"
        value = getattr(self, name, "max_hv")
        if not isinstance(value, str):
            raise ValueError("%s must be a string" % name)
        value = value.strip().lower()
        if value not in ("union", "max_hv", "max_hv_effective"):
            raise ValueError(
                "%s must be one of: union, max_hv, max_hv_effective" % name
            )
        self.gpugr_area_adjust_congestion_mode = value

    def normalize_params(self):
        self.normalize_removed_l_shape_weight_schedule()
        self.normalize_iopin_density_weight()
        self.normalize_m2_pg_rail_density_weight()
        self.normalize_m2_pg_rail_legalization_mode()
        self.normalize_m2_pg_rail_legalization_rail_range()
        self.normalize_m2_pg_rail_legalization_displacement_weight()
        self.normalize_m2_pa_refine_limits()
        self.normalize_post_legalization_adaptive_padding()
        self.normalize_legalize_before_each_inflation()
        self.normalize_inflation_area_budget_ratio()
        self.normalize_gpugr_area_adjust_congestion_mode()

    def fromJson(self, data):
        """
        @brief load from json
        """
        for key, value in data.items():
            self.__dict__[key] = value
        self.apply_l_shape_routability_preset()
        self.normalize_params()

    def dump(self, filename):
        """
        @brief dump to json file
        """
        with open(filename, "w") as f:
            json.dump(self.toJson(), f)

    def load(self, args):
        """
        @brief load parameters
        """
        if len(args) == 1 and args[0].endswith(".json"):
            with open(args[0], "r") as f:
                self.fromJson(json.load(f))
        else:
            self.fromCmdLine(args)

    def __str__(self):
        """
        @brief string
        """
        return str(self.toJson())

    def __repr__(self):
        """
        @brief print
        """
        return self.__str__()

    def __eq__(self, other):
        """
        @brief test equality
        """
        if isinstance(other, Params):
            ignore = {"params_dict"}
            return all(
                (self.__dict__[key] == other.__dict__[key]) | (key in ignore)
                for key in self.__dict__
            )
        return False

    def design_name(self):
        """
        @brief speculate the design name for dumping out intermediate solutions
        """    
        return self.base_design_name

    def solution_file_suffix(self):
        """
        @brief speculate placement solution file suffix
        """
        if self.def_input is not None and os.path.exists(self.def_input):  # LEF/DEF
            return "def"
        else:  # Bookshelf
            return "pl"

    def fromCmdLine(self, args):
        """
        @brief load from command line
        """
        for key, value in (
            (k.lstrip("--"), v) for k, v in (arg.split("=") for arg in args)
        ):
            self.__dict__[key] = value
        self.apply_l_shape_routability_preset()
        self.normalize_params()

    def update(self, params):
        """
        @brief update parameters
        """
        if isinstance(params, dict):
            self.fromJson(params)
        elif isinstance(params, str):
            self.load([params])
        elif isinstance(params, list):
            self.load(params)
        elif isinstance(params, Params):
            self.fromJson(params.__dict__)
