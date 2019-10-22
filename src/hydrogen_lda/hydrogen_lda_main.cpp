/*! \file hydrogen_lda_main.cpp
    \brief VWN-LDA汎関数を用い、Kohn-Sham法でヘリウム原子のエネルギーを計算する
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

#include "hydrogen_lda.h"
#include <iostream>             // for std::cerr, std::cout
#include <boost/format.hpp>     // for boost::format

int main()
{
    helium_lda::Helium_LDA hl;
    if (auto const res(hl.do_scfloop()); res) {
        std::cout << boost::format("SCF計算が収束しました: energy = %.14f (Hartree)") % (*res) << std::endl;

        return 0;
    }

    std::cerr << "SCF計算が収束しませんでした" << std::endl;
    return -1;
}
