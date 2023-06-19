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

/*
 Constructs the inner product circuit. num multiplications and num additions.
 */
share* BuildInnerProductCircuit(share *s_x, share *s_y, uint32_t numbers, ArithmeticCircuit *ac) {
	uint32_t i;

	// pairwise multiplication of all input values
	s_x = ac->PutMULGate(s_x, s_y);

	// split SIMD gate to separate wires (size many)
	s_x = ac->PutSplitterGate(s_x);

	// add up the individual multiplication results and store result on wire 0
	// in arithmetic sharing ADD is for free, and does not add circuit depth, thus simple sequential adding
	for (i = 1; i < numbers; i++) {
		s_x->set_wire_id(0, ac->PutADDGate(s_x->get_wire_id(0), s_x->get_wire_id(i)));
	}

	// discard all wires, except the addition result
	s_x->set_bitlength(1);

	return s_x;
}

void test_verilog_add64_SIMD(e_role role, const std::string& address, uint16_t port, seclvl seclvl, uint32_t nvals, uint32_t nthreads,
	e_mt_gen_alg mt_alg, e_sharing sharing, double afp, double bfp) {

	uint32_t bitlen = 32, numbers = 128, secparam = 128;
	/**
	 Step 1: Create the ABYParty object which defines the basis of all the
	 operations which are happening.	Operations performed are on the
	 basis of the role played by this object.
	 */
	ABYParty* party = new ABYParty(role, address, port, seclvl, bitlen, nthreads,
			mt_alg);

	/**
	 Step 2: Get to know all the sharing types available in the program.
	 */
	std::vector<Sharing*>& sharings = party->GetSharings();

	/**
	 Step 3: Create the circuit object on the basis of the sharing type
	 being inputed.
	 */
	ArithmeticCircuit* ac =
			(ArithmeticCircuit*) sharings[S_ARITH]->GetCircuitBuildRoutine();

	BooleanCircuit* bc =
			(BooleanCircuit*) sharings[S_BOOL]->GetCircuitBuildRoutine();
	
	Circuit* yc = sharings[S_YAO]->GetCircuitBuildRoutine();


	/**
	 Step 4: Creating the share objects - s_x_vec, s_y_vec which
	 are used as inputs to the computation. Also, s_out which stores the output.
	 */

	share *s_x_vec, *s_y_vec, *s_out;

	/**
	 Step 5: Allocate the xvals and yvals that will hold the plaintext values.
	 */
	uint32_t x, y;

	uint32_t output, v_sum = 0;

	std::vector<uint32_t> xvals(numbers);
	std::vector<uint32_t> yvals(numbers);

	uint32_t i;
	srand(time(NULL));

	/**
	 Step 6: Fill the arrays xvals and yvals with the generated random values.
	 Both parties use the same seed, to be able to verify the
	 result. In a real example each party would only supply
	 one input value. Copy the randomly generated vector values into the respective
	 share objects using the circuit object method PutINGate().
	 Also mention who is sharing the object.
	 The values for the party different from role is ignored,
	 but PutINGate() must always be called for both roles.
	 */
	for (i = 0; i < numbers; i++) {

		x = rand();
		y = rand();

		v_sum += x * y;

		xvals[i] = x;
		yvals[i] = y;
	}

	s_x_vec = ac->PutSIMDINGate(numbers, xvals.data(), bitlen, SERVER);
	s_y_vec = ac->PutSIMDINGate(numbers, yvals.data(), bitlen, CLIENT);

	/**
	 Step 7: Call the build method for building the circuit for the
	 problem by passing the shared objects and circuit object.
	 Don't forget to type cast the circuit object to type of share
	 */
	// s_out = BuildInnerProductCircuit(s_x_vec, s_y_vec, numbers,
	// 		(ArithmeticCircuit*) circ);

	// pairwise multiplication of all input values
	s_out = ac->PutMULGate(s_x_vec, s_y_vec);

	// split SIMD gate to separate wires (size many)
	s_out = ac->PutSplitterGate(s_out);

	// add up the individual multiplication results and store result on wire 0
	// in arithmetic sharing ADD is for free, and does not add circuit depth, thus simple sequential adding
	for (i = 1; i < numbers; i++) {
		s_out->set_wire_id(0, ac->PutADDGate(s_out->get_wire_id(0), s_out->get_wire_id(i)));
	}

	// discard all wires, except the addition result
	s_out->set_bitlength(1);

	ac->PutPrintValueGate(s_out,"s_out ac");

	s_out = bc->PutA2BGate(s_out, yc);

	bc->PutPrintValueGate(s_out,"s_out bc");

	//s_out = bc->PutConvTypeGate(s_out);

	s_out = bc->PutFPGate(s_out, SQRT);

	bc->PutPrintValueGate(s_out,"s_out sqrt");

	/**
	 Step 8: Output the value of s_out (the computation result) to both parties
	 */
	// s_out = ac->PutOUTGate(s_out, ALL); // AC
	s_out = bc->PutOUTGate(s_out, ALL);  // BC

	/**
	 Step 9: Executing the circuit using the ABYParty object evaluate the
	 problem.
	 */
	party->ExecCircuit();

	/**
	 Step 10: Type caste the plaintext output to 16 bit unsigned integer.
	 */

	output = s_out->get_clear_value<uint32_t>();
	std::cout << "\nCircuit Result: " << output;
	std::cout << "\nVerification Result: " << v_sum << std::endl;

	uint32_t *sqrt_out_vals = (uint32_t*) s_out->get_clear_value_ptr();

	float val = *((float*) sqrt_out_vals);

	std::cout << "SQRT RES: " << val << std::endl 
	<< "sum: " << v_sum << std::endl << 
	"sqrt(sum) = " << sqrt(static_cast<float>(v_sum)) << std::endl;

	delete s_x_vec;
	delete s_y_vec;
	delete party;

}



int main(int argc, char** argv) {

	e_role role;
	uint32_t bitlen = 32, nvals = 128, secparam = 128, nthreads = 1;

	uint16_t port = 7766;
	std::string address = "127.0.0.1";
	int32_t test_op = -1;
	e_mt_gen_alg mt_alg = MT_OT;
	uint32_t test_bit = 0;
	double fpa = 10.52, fpb = 1.30;

	read_test_options(&argc, &argv, &role, &bitlen, &nvals, &secparam, &address,
		&port, &test_op, &test_bit, &fpa, &fpb);

	std::cout << std::fixed << std::setprecision(3);

	seclvl seclvl = get_sec_lvl(secparam);


	test_verilog_add64_SIMD(role, address, port, seclvl, nvals, nthreads, mt_alg, S_ARITH, fpa, fpb);

	return 0;
}
