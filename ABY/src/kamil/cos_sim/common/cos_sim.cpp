/**
 \file 		innerproduct.cpp
 \author 	sreeram.sadasivam@cased.de
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
 \brief		Implementation of the Inner Product using ABY Framework.
 */

#include "cos_sim.h"
#include "../../../abycore/sharing/sharing.h"

int32_t test_inner_product_circuit(e_role role, const std::string& address, uint16_t port, seclvl seclvl,
		uint32_t numbers, uint32_t bitlen, uint32_t nthreads, e_mt_gen_alg mt_alg,
		e_sharing sharing) {

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
	ArithmeticCircuit* circ =
			(ArithmeticCircuit*) sharings[sharing]->GetCircuitBuildRoutine();

	/**
	 Step 4: Creating the share objects - s_x_vec, s_y_vec which
	 are used as inputs to the computation. Also, s_out which stores the output.
	 */

	share *s_x_vec, *s_y_vec, *s_out;

	/**
	 Step 5: Allocate the xvals and yvals that will hold the plaintext values.
	 */
	uint16_t x, y;

	uint16_t output, dot_x_y = 0, norm_x_wo_sqrt = 0, norm_y_wo_sqrt = 0;
	double norm_x = 0, norm_y = 0, cos_sim_x_y = 0;

	std::vector<uint16_t> xvals(numbers);
	std::vector<uint16_t> yvals(numbers);

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

		x = rand() % 100;
		y = rand() % 100;

		dot_x_y += x * y;
		norm_x_wo_sqrt += x * x;
		norm_y_wo_sqrt += y * y;

		xvals[i] = x;
		yvals[i] = y;
	}

	norm_x = sqrt(norm_x_wo_sqrt);
	norm_y = sqrt(norm_y_wo_sqrt);

	cos_sim_x_y = dot_x_y / (norm_x * norm_y);

	s_x_vec = circ->PutSIMDINGate(numbers, xvals.data(), 16, SERVER);
	s_y_vec = circ->PutSIMDINGate(numbers, yvals.data(), 16, CLIENT);

	/**
	 Step 7: Call the build method for building the circuit for the
	 problem by passing the shared objects and circuit object.
	 Don't forget to type cast the circuit object to type of share
	 */

	/* cos_sim = (x \dot y) / ( norm(x) * norm(y) ) */

	/* ------------- x \dot y ------------- */

	share* s_x_dot_y;

	// pairwise multiplication of all input values
	s_x_dot_y = circ->PutMULGate(s_x_vec, s_y_vec);

	// split SIMD gate to separate wires (size many)
	s_x_dot_y = circ->PutSplitterGate(s_x_dot_y);

	// add up the individual multiplication results and store result on wire 0
	// in arithmetic sharing ADD is for free, and does not add circuit depth, thus simple sequential adding
	for (i = 1; i < numbers; i++) {
		s_x_dot_y->set_wire_id(0, circ->PutADDGate(s_x_dot_y->get_wire_id(0), s_x_dot_y->get_wire_id(i)));
	}
	
	// discard all wires, except the addition result
	s_x_dot_y->set_bitlength(1);

	/* ------------- norm(x) ------------- */

	share* s_norm_x;

	// pairwise multiplication of all input values
	s_norm_x = circ->PutMULGate(s_x_vec, s_x_vec);

	// split SIMD gate to separate wires (size many)
	s_norm_x = circ->PutSplitterGate(s_norm_x);

	// add up the individual multiplication results and store result on wire 0
	// in arithmetic sharing ADD is for free, and does not add circuit depth, thus simple sequential adding
	for (i = 1; i < numbers; i++) {
		s_norm_x->set_wire_id(0, circ->PutADDGate(s_norm_x->get_wire_id(0), s_norm_x->get_wire_id(i)));
	}
	
	// discard all wires, except the addition result
	s_norm_x->set_bitlength(1);

	//todo SQRT

	/* ------------- norm(y) ------------- */

	share* s_norm_y;

	// pairwise multiplication of all input values
	s_norm_y = circ->PutMULGate(s_y_vec, s_y_vec);

	// split SIMD gate to separate wires (size many)
	s_norm_y = circ->PutSplitterGate(s_norm_y);

	// add up the individual multiplication results and store result on wire 0
	// in arithmetic sharing ADD is for free, and does not add circuit depth, thus simple sequential adding
	for (i = 1; i < numbers; i++) {
		s_norm_y->set_wire_id(0, circ->PutADDGate(s_norm_y->get_wire_id(0), s_norm_y->get_wire_id(i)));
	}
	
	// discard all wires, except the addition result
	s_norm_y->set_bitlength(1);

	//todo SQRT

	party->ExecCircuit();

	// Change to boolean circuit

	party->Reset();

	BooleanCircuit* bc =
			(BooleanCircuit*) sharings[S_BOOL]->GetCircuitBuildRoutine();

	



	s_out = bc->PutFPGate(s_norm_x, SQRT);

	/**
	 Step 8: Output the value of s_out (the computation result) to both parties
	 */
	s_out = circ->PutOUTGate(s_out, ALL);

	/**
	 Step 9: Executing the circuit using the ABYParty object evaluate the
	 problem.
	 */
	party->ExecCircuit();

	/**
	 Step 10: Type caste the plaintext output to 16 bit unsigned integer.
	 */
	//output = s_out->get_clear_value<uint16_t>();

	uint32_t *out_val = (uint32_t*) s_out->get_clear_value_ptr();
	float val = *((float*) out_val);

	std::cout << "\nCircuit Result: " << val << std::endl;
	std::cout << "\nVerification Results:";
	std::cout << "\nx = ";
	for (uint16_t i: xvals) std::cout << i << ", ";
	std::cout << "\ny = ";
	for (uint16_t i: yvals) std::cout << i << ", ";
	std::cout << "\ndot(x,y) = " << dot_x_y << std::endl;
	std::cout << "norm(x) w/o sqrt = " << norm_x_wo_sqrt << std::endl;
	std::cout << "norm(y) w/o sqrt = " << norm_y_wo_sqrt << std::endl;
	std::cout << "norm(x) = " << norm_x << std::endl;
	std::cout << "norm(y) = " << norm_y << std::endl;
	std::cout << "cos_sim(x,y) = " << cos_sim_x_y << std::endl;

	delete s_x_vec;
	delete s_y_vec;
	delete party;

	return 0;
}

/*
 Constructs the inner product circuit. num multiplications and num additions.
 */
share* BuildInnerProductCircuit(share *s_x, share *s_y, uint32_t numbers, ArithmeticCircuit *ac) {

	

	return NULL;
}
