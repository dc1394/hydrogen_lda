/*! \file gausslegendre.cpp
    \brief Gauss-Legendre積分を行うクラスの実装
    Copyright © 2019 @dc1394 All Rights Reserved.

    This program is free software; you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your option)
    any later version.

    This program is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
    more details.

    You should have received a copy of the GNU General Public License along
    with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "../../alglib/src/integration.h"	// for alglib
#include "gausslegendre.h"
#include <stdexcept>					    // for std::runtime_error
#include <boost/cast.hpp>				    // for boost::numeric_cast

namespace gausslegendre {
    // #region コンストラクタ

    Gauss_Legendre::Gauss_Legendre(std::int32_t n) : n_(n)
    {
        alglib::ae_int_t info = 0;
        alglib::real_1d_array x, w;

        alglib::gqgenerategausslegendre(
            boost::numeric_cast<alglib::ae_int_t>(n),
            info,
            x,
            w);

        if (info != 1) {
            throw std::runtime_error("alglib::gqgenerategausslegendreが失敗");
        }

        x_.assign(x.getcontent(), x.getcontent() + x.length());
        w_.assign(w.getcontent(), w.getcontent() + w.length());
    }

    // #endregion コンストラクタ
}

