# -*- coding: utf-8 -*-
"""

    tests._test_msui.test_remotesensing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This module provides pytest functions to tests msui.remotesensing module

    This file is part of MSS.

    :copyright: Copyright 2017 Reimar Bauer, Joern Ungermann
    :copyright: Copyright 2023 rootxrishabh
    :copyright: Copyright 2017-2023 by the MSS team, see AUTHORS.
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

from mslib.msui.remotesensing_dockwidget import RemoteSensingControlWidget
import skyfield_data
from PyQt5 import QtWidgets
from mslib.msui import mpl_qtwidget as qt
import datetime
import sys
from unittest.mock import Mock
import matplotlib as mpl
from matplotlib.collections import LineCollection
app = QtWidgets.QApplication(sys.argv)


def test_skyfield_data_expiration(recwarn):
    skyfield_data.check_expirations()
    assert len(recwarn) == 0, [_x.message for _x in recwarn]


class Test_RemoteSensingControlWidget(object):
    """
    Tests about RemoteSensingControlWidget
    """
    def setup_method(self):
        self.view = Mock()
        self.map = qt.TopViewPlotter()
        self.map.init_map()
        self.bmap = self.map.map
        self.result_test_direction_coordinates = [([79.083, 79.06123961616663, 79.03945385931264, 79.01764258716722,
                                                    78.99580565691319, 78.97394292518322, 78.95205424805584,
                                                    78.93013948105154, 78.9081984791288, 78.88623109668012,
                                                    78.864237187528, 78.84221660492089, 78.82016920152915,
                                                    78.79809482944094, 78.77599334015811, 78.75386458459202,
                                                    78.73170841305945, 78.7095246752783, 78.68731322036342,
                                                    78.66507389682234, 78.64280655255095, 78.62051103482925,
                                                    78.59818719031695, 78.57583486504909, 78.55345390443169,
                                                    78.53104415323725, 78.50860545560032, 78.48613765501302,
                                                    78.46364059432048, 78.4411141157163, 78.41855806073795,
                                                    78.39597227026216, 78.37335658450027, 78.35071084299352,
                                                    78.32803488460837, 78.30532854753174, 78.2825916692662,
                                                    78.25982408662517, 78.23702563572809, 78.21419615199551,
                                                    78.19133547014418, 78.1684434241821, 78.14551984740352,
                                                    78.12256457238392, 78.09957743097502, 78.07655825429953,
                                                    78.05350687274617, 78.03042311596444, 78.00730681285941,
                                                    77.98415779158651, 77.96097587954621, 77.93776090337876,
                                                    77.91451268895875, 77.89123106138983, 77.86791584499917,
                                                    77.84456686333206, 77.8211839391464, 77.79776689440706,
                                                    77.77431555028042, 77.75082972712865, 77.72730924450406,
                                                    77.70375392114339, 77.68016357496208, 77.6565380230484,
                                                    77.6328770816577, 77.60918056620646, 77.58544829126639,
                                                    77.56168007055848, 77.53787571694693, 77.51403504243316,
                                                    77.49015785814966, 77.46624397435384, 77.4422932004219,
                                                    77.4183053448425, 77.3942802152105, 77.3702176182207,
                                                    77.34611735966136, 77.32197924440783, 77.29780307641605,
                                                    77.27358865871601, 77.24933579340524, 77.2250442816421,
                                                    77.20071392363917, 77.1763445186565, 77.15193586499484,
                                                    77.12748775998881, 77.103],
                                                  [21.15, 21.236413253602002,
                                                   21.322822812946203, 21.409228659374225, 21.495630774175087,
                                                   21.582029138584748, 21.668423733785726, 21.754814540906672,
                                                   21.84120154102192, 21.927584715151085, 22.013964044258625,
                                                   22.1003395092534, 22.186711090988254, 22.27307877025955,
                                                   22.359442527806735, 22.445802344311907, 22.53215820039936,
                                                   22.618510076635104, 22.70485795352644, 22.79120181152149,
                                                   22.877541631008732, 22.96387739231652, 23.050209075712633,
                                                   23.136536661403802, 23.222860129535192, 23.30917946018998,
                                                   23.395494633388832, 23.481805629089404, 23.568112427185895,
                                                   23.654415007508497, 23.740713349822926, 23.827007433829923,
                                                   23.91329723916472, 23.999582745396552, 24.085863932028115,
                                                   24.172140778495084, 24.25841326416555, 24.344681368339515,
                                                   24.43094507024836, 24.517204349054293, 24.603459183849825,
                                                   24.68970955365722, 24.77595543742794, 24.862196814042093,
                                                   24.948433662307902, 25.034665960961092, 25.120893688664378,
                                                   25.207116824006864, 25.29333534550347, 25.379549231594343,
                                                   25.465758460644327, 25.5519630109423, 25.638162860700618,
                                                   25.724357988054535, 25.81054837106155, 25.896733987700873,
                                                   25.982914815872725, 26.069090833397787, 26.15526201801657,
                                                   26.241428347388744, 26.327589799092568, 26.4137463506242,
                                                   26.499897979397105, 26.58604466274135, 26.672186377903017,
                                                   26.75832310204348, 26.8444548122388, 26.930581485479028,
                                                   27.016703098667524, 27.1028196286203, 27.18893105206533,
                                                   27.27503734564184, 27.36113848589966, 27.447234449298474,
                                                   27.53332521220713, 27.619410750902954, 27.705491041570983,
                                                   27.79156606030328, 27.877635783098203, 27.963700185859643,
                                                   28.04975924439631, 28.135812934420976, 28.221861231549717,
                                                   28.30790411130114, 28.393941549095683, 28.479973520254756,
                                                   28.566])]

        self.lon_lin = [79.083, 79.06123961616663, 79.03945385931264, 79.01764258716722, 78.99580565691319,
                        78.97394292518322, 78.95205424805584, 78.93013948105154,
                        78.9081984791288, 78.88623109668012, 78.864237187528,
                        78.84221660492089, 78.82016920152915,
                        78.79809482944094, 78.77599334015811, 78.75386458459202,
                        78.73170841305945, 78.7095246752783, 78.68731322036342,
                        78.66507389682234, 78.64280655255095,
                        78.62051103482925, 78.59818719031695, 78.57583486504909,
                        78.55345390443169, 78.53104415323725, 78.50860545560032,
                        78.48613765501302, 78.46364059432048,
                        78.4411141157163, 78.41855806073795, 78.39597227026216,
                        78.37335658450027, 78.35071084299352, 78.32803488460837,
                        78.30532854753174, 78.2825916692662,
                        78.25982408662517, 78.23702563572809, 78.21419615199551,
                        78.19133547014418, 78.1684434241821, 78.14551984740352,
                        78.12256457238392, 78.09957743097502,
                        78.07655825429953, 78.05350687274617, 78.03042311596444,
                        78.00730681285941, 77.98415779158651, 77.96097587954621,
                        77.93776090337876, 77.91451268895875,
                        77.89123106138983, 77.86791584499917, 77.84456686333206,
                        77.8211839391464, 77.79776689440706, 77.77431555028042,
                        77.75082972712865, 77.72730924450406,
                        77.70375392114339, 77.68016357496208, 77.6565380230484,
                        77.6328770816577, 77.60918056620646, 77.58544829126639,
                        77.56168007055848, 77.53787571694693,
                        77.51403504243316, 77.49015785814966, 77.46624397435384,
                        77.4422932004219, 77.4183053448425, 77.3942802152105,
                        77.3702176182207, 77.34611735966136,
                        77.32197924440783, 77.29780307641605, 77.27358865871601,
                        77.24933579340524, 77.2250442816421, 77.20071392363917,
                        77.1763445186565, 77.15193586499484,
                        77.12748775998881, 77.103]

        self.lat_lin = [21.15, 21.236413253602002, 21.322822812946203, 21.409228659374225, 21.495630774175087,
                        21.582029138584748, 21.668423733785726, 21.754814540906672,
                        21.84120154102192, 21.927584715151085, 22.013964044258625,
                        22.1003395092534, 22.186711090988254,
                        22.27307877025955, 22.359442527806735, 22.445802344311907,
                        22.53215820039936, 22.618510076635104, 22.70485795352644,
                        22.79120181152149, 22.877541631008732,
                        22.96387739231652, 23.050209075712633, 23.136536661403802,
                        23.222860129535192, 23.30917946018998, 23.395494633388832,
                        23.481805629089404, 23.568112427185895,
                        23.654415007508497, 23.740713349822926, 23.827007433829923,
                        23.91329723916472, 23.999582745396552, 24.085863932028115,
                        24.172140778495084, 24.25841326416555,
                        24.344681368339515, 24.43094507024836, 24.517204349054293,
                        24.603459183849825, 24.68970955365722, 24.77595543742794,
                        24.862196814042093, 24.948433662307902,
                        25.034665960961092, 25.120893688664378, 25.207116824006864,
                        25.29333534550347, 25.379549231594343, 25.465758460644327,
                        25.5519630109423, 25.638162860700618,
                        25.724357988054535, 25.81054837106155, 25.896733987700873,
                        25.982914815872725, 26.069090833397787, 26.15526201801657,
                        26.241428347388744, 26.327589799092568,
                        26.4137463506242, 26.499897979397105, 26.58604466274135,
                        26.672186377903017, 26.75832310204348, 26.8444548122388,
                        26.930581485479028, 27.016703098667524,
                        27.1028196286203, 27.18893105206533, 27.27503734564184,
                        27.36113848589966, 27.447234449298474, 27.53332521220713,
                        27.619410750902954, 27.705491041570983,
                        27.79156606030328, 27.877635783098203, 27.963700185859643,
                        28.04975924439631, 28.135812934420976, 28.221861231549717,
                        28.30790411130114, 28.393941549095683,
                        28.479973520254756, 28.566]

        self.cut_height = 10.0
        self.result_test_tangent_point_coordinates = [(81.20427368859193, 21.619554254892783),
                                                      (81.18360180986066, 21.706244389806088),
                                                      (81.16291034390225, 21.79293221855446),
                                                      (81.1421991611211, 21.879617725969652),
                                                      (81.12146813144997, 21.966300896852562),
                                                      (81.1007171243466, 22.052981715971455),
                                                      (81.07994600878948, 22.139660168064314),
                                                      (81.05915465327473, 22.226336237836602),
                                                      (81.03834292581215, 22.313009909961885),
                                                      (81.01751069392179, 22.399681169080576),
                                                      (80.99665782462979, 22.48634999980179),
                                                      (80.97578418446517, 22.57301638669984),
                                                      (80.95488963945591, 22.6596803143158),
                                                      (80.93397405512486, 22.746341767157855),
                                                      (80.91303729648612, 22.833000729699883),
                                                      (80.8920792280416, 22.91965718638),
                                                      (80.87109971377654, 23.00631112160272),
                                                      (80.85009861715581, 23.09296251973775),
                                                      (80.82907580112028, 23.179611365117932),
                                                      (80.80803112808246, 23.26625764204181),
                                                      (80.78696445992283, 23.35290133477092),
                                                      (80.76587565798584, 23.439542427530032),
                                                      (80.74476458307562, 23.526180904507914),
                                                      (80.72363109545232, 23.61281674985503),
                                                      (80.70247505482736, 23.699449947686343),
                                                      (80.68129632036037, 23.78608048207612),
                                                      (80.66009475065403, 23.872708337062818),
                                                      (80.63887020375049, 23.959333496644938),
                                                      (80.61762253712718, 24.04595594478169),
                                                      (80.59635160769209, 24.132575665394203),
                                                      (80.5750572717801, 24.21919264236301),
                                                      (80.55373938514839, 24.30580685952867),
                                                      (80.53239780297216, 24.392418300691585),
                                                      (80.51103237984037, 24.479026949610862),
                                                      (80.48964296975147, 24.56563279000407),
                                                      (80.46822942610852, 24.65223580554839),
                                                      (80.44679160171529, 24.738835979877916),
                                                      (80.42532934877154, 24.82543329658476),
                                                      (80.40384251886852, 24.912027739218377),
                                                      (80.38233096298472, 24.998619291284196),
                                                      (80.3607945314808, 25.085207936245087),
                                                      (80.33923307409535, 25.171793657519576),
                                                      (80.31764643994016, 25.258376438481864),
                                                      (80.29603447749594, 25.344956262460002),
                                                      (80.27439703460642, 25.431533112740297),
                                                      (80.25273395847552, 25.51810697255911),
                                                      (80.23104509566079, 25.60467782510959),
                                                      (80.20933029206962, 25.691245653537543),
                                                      (80.18758939295407, 25.77781044094238),
                                                      (80.16582224290622, 25.864372170375102),
                                                      (80.14402868585302, 25.950930824839897),
                                                      (80.12220856505132, 26.03748638729289),
                                                      (80.10036172308344, 26.124038840640296),
                                                      (80.0784880018513, 26.21058816774082),
                                                      (80.0565872425721, 26.297134351402587),
                                                      (80.03465928577303, 26.383677374383492),
                                                      (80.01270397128556, 26.470217219393053),
                                                      (79.99072113824144, 26.55675386908735),
                                                      (79.96871062506634, 26.643287306073162),
                                                      (79.94667226947563, 26.729817512903203),
                                                      (79.92460590846814, 26.816344472080264),
                                                      (79.90251137832155, 26.902868166052826),
                                                      (79.88038851458653, 26.989388577217603),
                                                      (79.8582371520822, 27.075905687915036),
                                                      (79.83605712488925, 27.162419480434753),
                                                      (79.81384826634594, 27.2489299370091),
                                                      (79.7916104090419, 27.335437039815744),
                                                      (79.76934338481207, 27.421940770978352),
                                                      (79.74704702473245, 27.508441112561886),
                                                      (79.72472115911295, 27.59493804657693),
                                                      (79.70236561749279, 27.681431554975404),
                                                      (79.67998022863429, 27.76792161965211),
                                                      (79.65756482051724, 27.854408222443364),
                                                      (79.63511922033271, 27.94089134512787),
                                                      (79.6126432544781, 28.027370969422904),
                                                      (79.59013674854992, 28.113847076988744),
                                                      (79.56759952733928, 28.20031964942261),
                                                      (79.54503141482452, 28.286788668262982),
                                                      (79.52243223416616, 28.37325411498613),
                                                      (79.49980180770042, 28.459715971006098),
                                                      (79.47713995693299, 28.546174217675254),
                                                      (79.45444650253313, 28.632628836281985),
                                                      (79.43172126432714, 28.71907980805177),
                                                      (79.40896406129221, 28.80552711414579),
                                                      (79.38617471155024, 28.89197073566029),
                                                      (79.3633530323609, 28.978410653627137)]
        self.wp_vertices = [(0, 0), (1, 4)]
        self.wp_heights = [0, 1000]
        self.coordinates = [[79.083, 21.15], [77.103, 28.566]]
        self.heights = [0.0, 0.0]
        self.times = [datetime.datetime(2023, 4, 15, 10, 9, 59, 174000),
                      datetime.datetime(2023, 4, 15, 11, 18, 27, 735581)]
        self.solar_type = ('sun', 'total (horizon)')

    def test_view_angles(self):
        compute_view_angles = RemoteSensingControlWidget.compute_view_angles
        angle = compute_view_angles(0, 0, 0, 1, 0, 0, 0, -1)
        assert angle[0] == 90.0
        assert angle[1] == -1
        angle = compute_view_angles(0, 0, 0, -1, 0, 0, 0, -1)
        assert angle[0] == 270.0
        assert angle[1] == -1
        angle = compute_view_angles(0, 0, 0, 1, 0, 0, 90, -1)
        assert angle[0] == 180.0
        assert angle[1] == -1
        angle = compute_view_angles(0, 0, 0, 0, 1, 0, 0, -1)
        assert angle[0] == 0.0
        assert angle[1] == -1
        angle = compute_view_angles(0, 0, 0, 0, -1, 0, 0, -1)
        assert angle[0] == 180.0
        assert angle[1] == -1

    def test_body_angle(self):
        compute_body_angle = RemoteSensingControlWidget(view=self.view).compute_body_angle
        angle = compute_body_angle("sun", 734811473.5682117, 78.01244659053887, 25.274176073890143)
        assert angle[0] == 347.07153397468835
        assert angle[1] == -54.44327070161909
        angle = compute_body_angle("sun", 734811930.075054, 77.78995359959562, 26.0978150996952)
        assert angle[0] == 350.0832778892227
        assert angle[1] == -53.93384834862541
        angle = compute_body_angle("sun", 734812386.5818964, 77.56432276488127, 26.92101210457491)
        assert angle[0] == 353.01248647941395
        assert angle[1] == -53.33819642473756
        angle = compute_body_angle("sun", 734812843.0887387, 77.33539399373548, 27.743747257732803)
        assert angle[0] == 355.85087364088474
        assert angle[1] == -52.66164278537733
        angle = compute_body_angle("sun", 734811473.5682117, 78.01244659053887, 25.274176073890143)
        assert angle[0] == 347.07153397468835
        assert angle[1] == -54.44327070161909

    def test_direction_coordinates(self):
        compute_direction_coordinates = RemoteSensingControlWidget(view=self.view).direction_coordinates
        coordinates = compute_direction_coordinates(self.result_test_direction_coordinates)
        assert coordinates == [[(78.11107100167946, 24.905315238175), (79.1827251371111, 25.140314258548926)]]

    def test_compute_tangent_lines(self):

        result = RemoteSensingControlWidget(view=self.view).compute_tangent_lines(self.bmap,
                                                                                  self.wp_vertices, self.wp_heights)
        assert isinstance(result, LineCollection)
        assert len(result.get_segments()) == len(self.wp_heights)

    def test_compute_solar_lines(self):
        result = RemoteSensingControlWidget(view=self.view)
        result = result.compute_solar_lines(self.bmap, self.coordinates, self.heights, self.times, self.solar_type)
        assert isinstance(result, mpl.collections.LineCollection)

    def test_tangent_point_coordinates(self):
        tangent_point_coordinates = RemoteSensingControlWidget(view=self.view).tangent_point_coordinates
        coordinates = tangent_point_coordinates(lon_lin=self.lon_lin, lat_lin=self.lat_lin, cut_height=self.cut_height)
        assert coordinates == self.result_test_tangent_point_coordinates

    def test_calc_view_rating(self):
        height = 0.0
        difftype = "total (horizon)"
        calc_view_rating = RemoteSensingControlWidget(view=self.view).calc_view_rating
        view_rating = calc_view_rating(obs_azi=76.00751678328697, obs_ele=-1.0, sol_azi=240.70922332385769,
                                       sol_ele=58.33227874190401, height=height, difftype=difftype)
        assert view_rating == 175.06276428208045

        view_rating = calc_view_rating(obs_azi=76.11546709722984, obs_ele=-1.0, sol_azi=239.9012770471841,
                                       sol_ele=60.030135582543686, height=height, difftype=difftype)
        assert view_rating == 174.78692454009882

        view_rating = calc_view_rating(obs_azi=76.50369446610219, obs_ele=-1.0, sol_azi=236.15310358444265,
                                       sol_ele=66.92788781559761, height=height, difftype=difftype)
        assert view_rating == 173.49965929339362
        
