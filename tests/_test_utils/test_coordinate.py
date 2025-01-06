# -*- coding: utf-8 -*-
"""

    tests._test_utils.test_coordinate
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This module provides pytest functions to tests mslib.utils.coordinate

    This file is part of MSS.

    :copyright: Copyright 2016-2017 Reimar Bauer
    :copyright: Copyright 2016-2024 by the MSS team, see AUTHORS.
    :license: APACHE-2.0, see LICENSE for details.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import logging
import datetime
import numpy as np
import pytest

import mslib.utils.coordinate as coordinate
from mslib.utils.find_location import find_location
from mslib.utils.get_projection_params import get_projection_params

LOGGER = logging.getLogger(__name__)


class TestGetDistance:
    """
    Tests for distance-based calculations in `coordinate` module.
    """

    @pytest.mark.parametrize("lat0, lon0, lat1, lon1, expected_distance", [
        (50.355136, 7.566077, 50.353968, 4.577915, 212),
        (-5.135943, -42.792442, 4.606085, 120.028077, 18130),
    ])
    def test_get_distance(self, lat0, lon0, lat1, lon1, expected_distance):
        """
        Test the calculation of distances between coordinate pairs.
        """
        assert int(coordinate.get_distance(lat0, lon0, lat1, lon1)) == expected_distance

    @pytest.mark.parametrize("lat, lon, expected_location", [
        (50.92, 6.36, ([50.92, 6.36], 'Juelich')),
        (50.9200002, 6.36, ([50.92, 6.36], 'Juelich')),
    ])
    def test_find_location(self, lat, lon, expected_location):
        """
        Test finding the location from coordinates.
        """
        assert find_location(lat, lon) == expected_location


class TestProjections:
    """
    Tests for handling coordinate projections.
    """

    def test_get_projection_params(self):
        """
        Test fetching projection parameters for various EPSG codes.
        """
        assert get_projection_params("epsg:4839") == {'basemap': {'epsg': '4839'}, 'bbox': 'meter(10.5,51)'}
        for invalid_code in ['auto2:42005', 'auto:42001', 'crs:83']:
            with pytest.raises(ValueError):
                get_projection_params(invalid_code)


class TestAngles:
    """
    Tests for angle normalization and point rotation.
    """

    @pytest.mark.parametrize("angle, normalized", [
        (0, 0),
        (180, 180),
        (270, 270),
        (-90, 270),
        (-180, 180),
        (-181, 179),
        (420, 60),
    ])
    def test_normalize_angle(self, angle, normalized):
        """
        Test normalizing angles to the range [0, 360).
        """
        assert coordinate.fix_angle(angle) == normalized

    @pytest.mark.parametrize("point, angle, rotated_point", [
        ([0, 0], 0, (0.0, 0.0)),
        ([0, 0], 180, (0.0, 0.0)),
        ([1, 0], 0, (1.0, 0.0)),
        ([100, 90], 90, (-90.0, 100.0)),
    ])
    def test_rotate_point(self, point, angle, rotated_point):
        """
        Test rotating points around the origin.
        """
        assert coordinate.rotate_point(point, angle) == rotated_point


class TestLatLonPoints:
    """
    Tests for generating lat/lon points along paths.
    """

    @pytest.mark.parametrize("ref_lats, ref_lons, numpoints, connection, expected_lats, expected_lons", [
        ([0, 10], [0, 0], 2, "linear", [0, 10], [0, 0]),
        ([0, 10], [0, 0], 3, "linear", [0, 5, 10], [0, 0, 0]),
        ([0, 0], [0, 10], 3, "linear", [0, 0, 0], [0, 5, 10]),
    ])
    def test_linear(self, ref_lats, ref_lons, numpoints, connection, expected_lats, expected_lons):
        """
        Test generating linear lat/lon points.
        """
        lats, lons = coordinate.latlon_points(ref_lats[0], ref_lons[0], ref_lats[1], ref_lons[1],
                                              numpoints=numpoints, connection=connection)
        np.testing.assert_array_equal(lats, expected_lats)
        np.testing.assert_array_equal(lons, expected_lons)

    @pytest.mark.parametrize("ref_lats, ref_lons, numpoints", [
        ([0, 10], [0, 0], 2),
        ([0, 10], [0, 0], 3),
    ])
    def test_greatcircle(self, ref_lats, ref_lons, numpoints):
        """
        Test generating lat/lon points along a great circle path.
        """
        lats, lons = coordinate.latlon_points(ref_lats[0], ref_lons[0], ref_lats[1], ref_lons[1],
                                              numpoints=numpoints, connection="greatcircle")
        assert len(lats) == numpoints
        assert len(lons) == numpoints


def test_pathpoints():
    """
    Test generating path points with timestamps for different connections.
    """
    lats = [0, 10]
    lons = [0, 10]
    times = [datetime.datetime(2012, 7, 1, 10, 30),
             datetime.datetime(2012, 7, 1, 10, 40)]
    ref = [lats, lons, times]

    for numpoints, connection in [(100, "linear"), (100, "greatcircle"), (200, "linear"), (200, "greatcircle")]:
        result = coordinate.path_points(lats, lons, numpoints, times=times, connection=connection)
        assert all(len(_x) == numpoints for _x in result)
        for i in range(3):
            assert pytest.approx(result[i][0]) == ref[i][0]
            assert pytest.approx(result[i][-1]) == ref[i][-1]

    lats = [0, 10, -20]
    lons = [0, 10, 20]
    times = [datetime.datetime(2012, 7, 1, 10, 30),
             datetime.datetime(2012, 7, 1, 10, 40),
             datetime.datetime(2012, 7, 1, 10, 50)]
    ref = [lats, lons, times]

    result = coordinate.path_points(lats, lons, 100, times=times, connection="linear")
    assert all(len(_x) == 100 for _x in result)
    for i in range(3):
        assert pytest.approx(result[i][0]) == ref[i][0]
        assert pytest.approx(result[i][-1]) == ref[i][-1]
