/*! \file helium_lda.h
    \brief VWN-LDAを用い、Kohn-Sham法でヘリウム原子のエネルギーを計算するクラスの宣言
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

#ifndef _HELIUM_LDA_H_
#define _HELIUM_LDA_H_

#pragma once

#include "gausslegendre/gausslegendre.h"
#include <cstdint>                      // for std::int32_t
#include <memory>                       // for std::shared_ptr, std::unique_ptr
#include <optional>                     // for std::optional
#include <valarray>                     // for std::valarray
#include <boost/multi_array.hpp>        // for boost::multi_array
#include <Eigen/Core>                   // for Eigen::MatrixXd, Eigen::VectorXd
#include <xc.h>                         // for xc_func_end

namespace hydrogen_lda {
    //! A lambda expression.
    /*!
        xc_func_typeへのポインタを解放するラムダ式
        \param xcfunc xc_func_type へのポインタ
    */
    static auto const xcfunc_deleter = [](auto * xcfunc) {
        xc_func_end(xcfunc);
        delete xcfunc;
    };

    //! A class.
    /*!
        VWN-LDAを用い、Kohn-Sham法でヘリウム原子のエネルギーを計算するクラス
    */
    class Hydrogen_LDA final {
        // #region コンストラクタ・デストラクタ

    public:
        //! A constructor.
        /*!
            唯一のコンストラクタ
        */
        Hydrogen_LDA();

        //! A destructor.
        /*!
            デストラクタ
        */
        ~Hydrogen_LDA() = default;

        // #region publicメンバ関数

        //! A public member function.
        /*!
            SCF計算を行う
            \return SCF計算が正常に終了した場合はエネルギーを、しなかった場合はstd::nulloptを返す
        */
        std::optional<double> do_scfloop();

		//! A private member function (const).
		/*!
			エネルギーの内訳を表示する
		*/
		void express_energy_breakdown() const;

        // #endregion publicメンバ関数

        // #region privateメンバ関数

    private:
        //! A private member function.
        /*!
            nalpha個のGTOによるヘリウム原子のエネルギーを計算する
            \return ヘリウム原子のエネルギー
        */
        double calc_energy();

        //! A private member function (const).
        /*!
            nalpha個のGTOによるヘリウム原子の交換相関エネルギーを計算する
            \return ヘリウム原子の交換相関エネルギー
        */
        double calc_exc_energy() const;
    	
        //! A private member function.
        /*!
            使用するGTOの数をユーザに入力させる
        */
        void input_nalpha();

        //! A private member function.
        /*!
            GTOの肩の係数が格納された配列を生成する
        */
        void make_alpha();

        //! A private member function.
        /*!
            全ての要素が、引数で指定された値で埋められたnalpha次元ベクトルを生成する
            \param val 要素を埋める値
        */
        void make_c(double val);

        //! A private member function.
        /*!
            交換相関積分が格納された、nalpha×nalphaの2次元配列を生成する
        */
        void make_exchcorrinteg();

        //! A private member function.
        /*!
            nalphaの数で、固有ベクトル、1電子積分および2電子積分からFock行列を生成する
        */
        void make_fockmatrix();

        //! A private member function.
        /*!
            1電子積分が格納された、nalpha×nalphaの2次元配列を生成する
        */
        void make_oneelectroninteg();

        //! A private member function.
        /*!
            nalpha次正方行列の重なり行列を生成する
        */
        void make_overlapmatrix();

        //! A private member function.
        /*!
            2電子積分が格納されたnalpha×nalpha×nalpha×nalphaの4次元配列を生成する
        */
        void make_twoelectroninteg();

        //! A private member function.
        /*!
            固有ベクトルを正規化する
        */
        void normalize();

        // #endregion privateメンバ関数

        // #region メンバ変数

#ifdef _DEBUG
		//! A private member variable (constant expression).
		/*!
			許容誤差
		*/
		static auto constexpr EPS = 1.0E-13;
#endif
  
        //! A private member variable (constant expression).
        /*!
            Gauss-Legendre積分の分点
        */
        static auto constexpr INTEGTABLENUM = 100;

        //! A private member variable (constant expression).
        /*!
            バッファサイズの上限
        */
        static auto constexpr MAXBUFSIZE = 32;

        //! A private member variable (constant expression).
        /*!
            SCF計算のループの上限
        */
        static auto constexpr MAXITER = 1000;

        //! A private member variable (constant expression).
        /*!
            積分区間の上限
        */
        static auto constexpr MAXR = 10.0;

        //! A private member variable (constant expression).
        /*!
            SCF計算のループから抜ける際のエネルギーの差の閾値
        */
        static auto constexpr SCFTHRESHOLD = 1.0E-15;

        //! A private member variable.
        /*!
            GTOの肩の係数が格納されたstd::vector
        */
        std::valarray<double> alpha_;

        //! A private member variable.
        /*!
            固有ベクトルC
        */
        Eigen::VectorXd c_;

		//! A private member variable.
		/*!
			軌道エネルギーε
		*/
		double epsilon_ = 0.0;
    	
        //! A private member variable.
        /*!
            Fock行列
        */
        Eigen::MatrixXd f_;
        
        //! A private member variable.
        /*!
            Gauss-Legendre積分用オブジェクト
        */
        gausslegendre::Gauss_Legendre gl_;

        //! A private member variable.
        /*!
            1電子積分が格納された2次元配列
        */
        boost::multi_array<double, 2> h_;

        //! A private member variable.
        /*!
            交換相関積分が格納された2次元配列
        */
        boost::multi_array<double, 2> k_;

        //! A private member variable.
        /*!
            使用するGTOの数
        */
        std::int32_t nalpha_ = 0;

        //! A private member variable (constant).
        /*!
            相関汎関数へのスマートポインタ
        */
        std::shared_ptr<xc_func_type> const pcfunc_;

        //! A private member variable (constant).
        /*!
            交換汎関数へのスマートポインタ
        */
        std::shared_ptr<xc_func_type> const pxfunc_;

        //! A private member variable.
        /*!
            2電子積分が格納された4次元配列
        */
        boost::multi_array<double, 4> q_;

        //! A private member variable.
        /*!
            重なり行列
        */
        Eigen::MatrixXd s_;

        // #region 禁止されたコンストラクタ・メンバ関数

    public:
        //! A public copy constructor (deleted).
        /*!
            コピーコンストラクタ（禁止）
            \param dummy コピー元のオブジェクト（未使用）
        */
        Hydrogen_LDA(Hydrogen_LDA const & dummy) = delete;

        //! A public member function (deleted).
        /*!
            operator=()の宣言（禁止）
            \param dummy コピー元のオブジェクト（未使用）
            \return コピー元のオブジェクト
        */
        Hydrogen_LDA & operator=(Hydrogen_LDA const & dummy) = delete;

        // #endregion 禁止されたコンストラクタ・メンバ関数
    };
        
}

#endif  // HELIUM_LDA_H_
