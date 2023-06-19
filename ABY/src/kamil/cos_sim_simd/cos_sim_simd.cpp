/**
 \file 		abyfloat.cpp
 \author	daniel.demmler@ec-spride.de
 \copyright	ABY - A Framework for Efficient Mixed-protocol Secure Two-party Computation
 Copyright (C) 2019 Engineering Cryptographic Protocols Group, TU Darmstadt
			This program is free software: you can redistribute it and/or modify
            it under the terms of the GNU Lesser General Public License as published
            by the Free Software Foundation, either version 3 of the License, or
            (at your option) any later version.
            ABY is distributed in the hope that it will be useful,
            but WITHOUT ANY WARRANTY; without even the implied warranty of
            MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
            GNU Lesser General Public License for more details.
            You should have received a copy of the GNU Lesser General Public License
            along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <ENCRYPTO_utils/crypto/crypto.h>
#include <ENCRYPTO_utils/parse_options.h>
#include "../../abycore/aby/abyparty.h"
#include "../../abycore/circuit/share.h"
#include "../../abycore/circuit/booleancircuits.h"
#include "../../abycore/circuit/arithmeticcircuits.h"
#include "../../abycore/sharing/sharing.h"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <random>

void read_test_options(int32_t* argcp, char*** argvp, e_role* role,
	uint32_t* bitlen, uint32_t* nvals, uint32_t* secparam, std::string* address,
	uint16_t* port, int32_t* test_op, uint32_t* test_bit, double* fpa, double* fpb) {

	uint32_t int_role = 0, int_port = 0, int_testbit = 0;

	parsing_ctx options[] =
	{ {(void*) &int_role, T_NUM, "r", "Role: 0/1", true, false },
	{(void*) &int_testbit, T_NUM, "i", "test bit", false, false },
	{(void*) nvals, T_NUM, "n",	"Number of parallel operation elements", false, false },
	{(void*) bitlen, T_NUM, "b", "Bit-length, default 32", false,false },
	{(void*) secparam, T_NUM, "s", "Symmetric Security Bits, default: 128", false, false },
	{(void*) address, T_STR, "a", "IP-address, default: localhost", false, false },
	{(void*) &int_port, T_NUM, "p", "Port, default: 7766", false, false },
	{(void*) test_op, T_NUM, "t", "Single test (leave out for all operations), default: off", false, false },
	{(void*) fpa, T_DOUBLE, "x", "FP a", false, false },
	{(void*) fpb, T_DOUBLE, "y", "FP b", false, false }

	};

	if (!parse_options(argcp, argvp, options,
		sizeof(options) / sizeof(parsing_ctx))) {
		print_usage(*argvp[0], options, sizeof(options) / sizeof(parsing_ctx));
	std::cout << "Exiting" << std::endl;
	exit(0);
	}

	assert(int_role < 2);
	*role = (e_role) int_role;

	if (int_port != 0) {
		assert(int_port < 1 << (sizeof(uint16_t) * 8));
		*port = (uint16_t) int_port;
	}

	*test_bit = int_testbit;
}

void test_verilog_add64_SIMD(e_role role, const std::string& address, uint16_t port, seclvl seclvl, uint32_t nvals, uint32_t nthreads,
	e_mt_gen_alg mt_alg, e_sharing sharing, double afp, double bfp) {

	// for addition we operate on doubles, so set bitlen to 64 bits
	uint32_t bitlen = 64;

	std::string circuit_dir = "../../bin/circ/";

	ABYParty* party = new ABYParty(role, address, port, seclvl, bitlen, nthreads, mt_alg, 100000, circuit_dir);

	std::vector<Sharing*>& sharings = party->GetSharings();

	BooleanCircuit* bc = (BooleanCircuit*) sharings[S_BOOL]->GetCircuitBuildRoutine();
	ArithmeticCircuit* ac = (ArithmeticCircuit*) sharings[S_ARITH]->GetCircuitBuildRoutine();
	Circuit* yc = sharings[S_YAO]->GetCircuitBuildRoutine();

	
	// create two arrays of random doubles
	uint64_t xvals[nvals];
	uint64_t yvals[nvals];

	// init for random values
	double lower_bound = 0;
   	double upper_bound = 10;
   	std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
   	std::default_random_engine re;


	for (uint32_t i = 0; i < nvals; i++) {
		double random_x = unif(re);
		double random_y = unif(re);

		uint64_t *xptr = (uint64_t*) &random_x;
		uint64_t *yptr = (uint64_t*) &random_y;

		xvals[i] = *xptr;
		yvals[i] = *yptr;

	}


	// SIMD input gates
	share* s_xin = bc->PutSIMDINGate(nvals, xvals, bitlen, SERVER);
	share* s_yin = bc->PutSIMDINGate(nvals, yvals, bitlen, CLIENT);

	share *s_product = bc->PutFPGate(s_xin, s_yin, MUL, bitlen, nvals, no_status);

	//bc->PutPrintValueGate(s_product, "s_product");

	share *s_product_split = bc->PutSplitterGate(s_product);

	bc->PutPrintValueGate(s_product_split, "s_product_split");


	share *s_product_added = s_product_split->get_wire_ids_as_share(0);

	bc->PutPrintValueGate(s_product_added, "First wire");

	share *s_next_wire;

	for (int i = 1; i<nvals; i++) {
		s_next_wire = s_product_split->get_wire_ids_as_share(i);
		//s_product_added = bc->PutFPGate(s_product_added, s_next_wire, ADD, bitlen, nvals, no_status);
		bc->PutPrintValueGate(s_next_wire, "next wire");
	}

	share* s_product_out = bc->PutOUTGate(s_product_added, ALL);

	// // testing

	// share *s_x;

	// s_x = ac->PutB2AGate(s_product);

	// // split SIMD gate to separate wires (size many)
	// s_x = ac->PutSplitterGate(s_x);

	// // add up the individual multiplication results and store result on wire 0
	// // in arithmetic sharing ADD is for free, and does not add circuit depth, thus simple sequential adding
	// for (int i = 1; i < nvals; i++) {
	// 	s_x->set_wire_id(0, ac->PutADDGate(s_x->get_wire_id(0), s_x->get_wire_id(i)));
	// }

	// // discard all wires, except the addition result
	// s_x->set_bitlength(1);

	// share *s_product_out = ac->PutOUTGate(s_x, ALL);


	// share *s_product_test = bc->PutSplitterGate(s_product);
	// share *s_temp = bc->PutFPGate(s_product_test->get_wire_ids_as_share(0), s_product_test->get_wire_ids_as_share(1), ADD, bitlen, nvals, no_status);
	// bc->PutPrintValueGate (s_temp,"Temp share\n");
	// for (int i = 1; i < nvals; i++) {
	// share *s_temp = circ->PutFPGate(s_product_test->get_wire_ids_as_share(0), s_product_test->get_wire_ids_as_share(i), ADD, bitlen, nvals, no_status);
	// circ->PutPrintValueGate (s_temp,"Temp share\n");
	// // 	//s_product_test->set_wire_id(0, );
	// }

	party->ExecCircuit();

	// retrieve plantext output
	uint32_t out_bitlen_product, out_nvals;
	uint64_t *out_vals_product;

	// s_product_out->get_clear_value_vec(&out_vals_product, &out_bitlen_product, &out_nvals);

	// // printing result
	// for (uint32_t i = 0; i < nvals; i++) {
	// 	// dereference output value as double without casting the content
	// 	double val = *((double*) &out_vals_product[i]);
	// 	std::cout << "product: " << val << " = " << *(double*) &xvals[i] << " * " << *(double*) &yvals[i] << std::endl;
	// }

	uint32_t *sqrt_out_vals = (uint32_t*) s_product_out->get_clear_value_ptr();

	double val = *((double*) sqrt_out_vals);

	std::cout << "DOT_PRODUCT: " << val << std::endl;

}


int main(int argc, char** argv) {

	e_role role;
	uint32_t bitlen = 64, nvals = 128, secparam = 128, nthreads = 1;

	uint16_t port = 7766;
	std::string address = "127.0.0.1";
	int32_t test_op = -1;
	e_mt_gen_alg mt_alg = MT_OT;
	uint32_t test_bit = 0;
	double fpa = 10.52, fpb = 1.30;

	read_test_options(&argc, &argv, &role, &bitlen, &nvals, &secparam, &address,
		&port, &test_op, &test_bit, &fpa, &fpb);

	std::cout << std::fixed << std::setprecision(3);
	std::cout << "double input values: " << fpa << " ; " << fpb << std::endl;

	seclvl seclvl = get_sec_lvl(secparam);


	test_verilog_add64_SIMD(role, address, port, seclvl, nvals, nthreads, mt_alg, S_BOOL, fpa, fpb);

	return 0;
}
