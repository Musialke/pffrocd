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
#include <fstream>




void read_test_options(int32_t *argcp, char ***argvp, e_role *role,
					   uint32_t *bitlen, uint32_t *nvals, uint32_t *secparam, std::string *address,
					   uint16_t *port, int32_t *test_op, uint32_t *test_bit, e_mt_gen_alg *mt_alg, uint32_t *debug, std::string *inputfile, std::string *pffrocd_path)
{

	uint32_t int_role = 0, int_port = 0, int_testbit = 0, int_mt_alg = 0;

	parsing_ctx options[] =
		{{(void *)&int_role, T_NUM, "r", "Role: 0/1", true, false},
		 {(void *)&int_testbit, T_NUM, "i", "test bit", false, false},
		 {(void *)nvals, T_NUM, "n", "Number of parallel operation elements", false, false},
		 {(void *)bitlen, T_NUM, "b", "Bit-length, default 32", false, false},
		 {(void *)secparam, T_NUM, "s", "Symmetric Security Bits, default: 128", false, false},
		 {(void *)address, T_STR, "a", "IP-address, default: localhost", false, false},
		 {(void *)&int_port, T_NUM, "p", "Port, default: 7766", false, false},
		 {(void *)test_op, T_NUM, "t", "Single test (leave out for all operations), default: off", false, false},
		 {(void *)&int_mt_alg, T_NUM, "x", "Arithmetic multiplication triples algorithm", false, false},
		 {(void *)debug, T_NUM, "d", "debug mode (more printing) (0/1)", false, false},
		 {(void *)inputfile, T_STR, "f", "Input file containing face embeddings", true, false},
		 {(void *)pffrocd_path, T_STR, "o", "absolute path to pffrocd directory", true, false}
		};
	std::cout << "Start pre checks" << std::endl; 
	if (!parse_options(argcp, argvp, options,
					   sizeof(options) / sizeof(parsing_ctx)))
	{
		print_usage(*argvp[0], options, sizeof(options) / sizeof(parsing_ctx));
		std::cout << "Exiting" << std::endl;
		exit(0);
	}

	assert(int_role < 2);
	*role = (e_role)int_role;

	if (int_port != 0)
	{
		assert(int_port < 1 << (sizeof(uint16_t) * 8));
		*port = (uint16_t)int_port;
	}

	*test_bit = int_testbit;

	if (int_mt_alg == 0) {
		*mt_alg = MT_OT;
	} else if (int_mt_alg == 1) {
		*mt_alg = MT_PAILLIER;
	} else if (int_mt_alg == 2) {
		*mt_alg = MT_DGK;
	} else {
		std::cout << "Invalid MT algorithm" << std::endl;
		exit(EXIT_FAILURE);
	}
	std::cout << "Finished pre checks" << std::endl; 
}

share* BuildInnerProductCircuit(share *s_x, share *s_y, uint32_t numbers, ArithmeticCircuit *ac) {
	uint32_t i;

	// pairwise multiplication of all input values
	share *dot_xy = ac->PutMULGate(s_x, s_y);

	// split SIMD gate to separate wires (size many)
	dot_xy = ac->PutSplitterGate(dot_xy);

	// add up the individual multiplication results and store result on wire 0
	// in arithmetic sharing ADD is for free, and does not add circuit depth, thus simple sequential adding
	for (i = 1; i < numbers; i++) {
		dot_xy->set_wire_id(0, ac->PutADDGate(dot_xy->get_wire_id(0), dot_xy->get_wire_id(i)));
	}

	// discard all wires, except the addition result
	dot_xy->set_bitlength(1);

	return dot_xy;
}

void test_verilog_add64_SIMD(e_role role, const std::string &address, uint16_t port, seclvl seclvl, uint32_t nvals, uint32_t nthreads,
							 e_mt_gen_alg mt_alg, e_sharing sharing, uint32_t debug, std::string inputfile, std::string pffrocd_path)
{
	std::cout << "Start of test" << std::endl; 
	// std::cout << "SEC LEVEL: " << seclvl.symbits << std::endl;
	// std::cout << "MT_ALG: " << mt_alg << std::endl;

	// 32 bit version operating on floats
	uint32_t bitlen = 32;

	// two arrays of real-world embeddings
	std::vector<int32_t> xembeddings;
	std::vector<int32_t> yembeddings;

	// array for the Sy<role> share
	std::vector<int32_t> share_embeddings;
	std::vector<int32_t> share_embeddings_prime;
	std::vector<float> share_scalar_x;
	std::vector<float> share_scalar_y;


	// reading the non-xored embeddings, i.e. current face and database face

	std::fstream infile(inputfile);

	// std::cout << "INPUT FILE NAME: " << inputfile << std::endl;

	int32_t x, y;

	// std::cout << "starting reading x and y" << std::endl;

	while (infile >> x >> y) {
		// std::cout << "x: " << x << " | y: "<< y << std::endl;
		xembeddings.push_back(x);
		yembeddings.push_back(y);
	}

	// std::cout<<"finished reading x and y" << std::endl;

	assert(xembeddings.size() == nvals);
	assert(yembeddings.size() == nvals);

	// std::cout << "nvals" << nvals << std::endl;
	// std::cout << "xembeddings: " << xembeddings.size() << std::endl;

	// // reading the xored embedding, i.e. either Sy<0> or Sy<1> depending on the role

	// // char *fname = (char *) malloc(150); // file name buffer 
    // // sprintf(fname, "/home/dietpi/pffrocd/ABY/build/bin/share%d.txt", role);

	std::string fname = pffrocd_path + "/ABY/build/bin/share" + std::to_string(role) + ".txt";
	std::string fnameprime = pffrocd_path + "/ABY/build/bin/share" + std::to_string(role) + "prime.txt";
	std::string fname_scalar_sharex = pffrocd_path + "/ABY/build/bin/share" + std::to_string(role) + "scalar_x.txt";
	std::string fname_scalar_sharey = pffrocd_path + "/ABY/build/bin/share" + std::to_string(role) + "scalar_y.txt";
	// //std::cout << "FNAME: " << fname << std::endl; 

	std::fstream infile_share(fname);
	std::fstream infile_share_prime(fnameprime);
	std::fstream infile_scalar_sharex(fname_scalar_sharex);
	std::fstream infile_scalar_sharey(fname_scalar_sharey);

	uint32_t z;
	float q;
	// // std::cout << "starting reading z" << std::endl;

	while(infile_share >> z) {
		//std::cout << "z: " << z << std::endl;
		share_embeddings.push_back(z);
	}
	z = 0;
	while(infile_share_prime >> z) {
		//std::cout << "z: " << z << std::endl;
		share_embeddings_prime.push_back(z);
	}
	q = 0;
	while(infile_scalar_sharex >> q) {
		//std::cout << "z: " << z << std::endl;
		share_scalar_x.push_back(q);
	}
	q = 0;
	while(infile_scalar_sharey >> q) {
		//std::cout << "z: " << z << std::endl;
		share_scalar_y.push_back(q);
	}

	// //std::cout<<"finished reading z" << std::endl;
	// std::cout << "share_embeddings size: " << share_embeddings.size() << std::endl;
	// std::cout << "share embedddings 0: " << share_embeddings[0] << std::endl;
	// assert(share_embeddings.size() == nvals);
	// assert(share_embeddings_prime.size() == nvals);

	std::string circuit_dir = pffrocd_path + "/ABY/bin/circ/";

	std::cout << "CIRCUIT DIRECTORY: " << circuit_dir << std::endl;

	ABYParty *party = new ABYParty(role, address, port, seclvl, bitlen, nthreads, mt_alg, 100000, circuit_dir);

	std::cout << "party created" << std::endl;


	std::vector<Sharing *> &sharings = party->GetSharings();

	BooleanCircuit *bc = (BooleanCircuit *)sharings[S_BOOL]->GetCircuitBuildRoutine();
	ArithmeticCircuit *ac = (ArithmeticCircuit *)sharings[S_ARITH]->GetCircuitBuildRoutine();
	Circuit *yc = (BooleanCircuit *)sharings[S_YAO]->GetCircuitBuildRoutine();

	/**
	 Step 4: Creating the share objects - s_x_vec, s_y_vec which
	 are used as inputs to the computation. Also, s_out which stores the output.
	 */

	share *s_x_vec, *s_y_vec, *s_out, *s_out_scalar;

	/**
	 Step 5: Allocate the xvals and yvals that will hold the plaintext values.
	 */
	int32_t output, v_sum = 0;
	float output_scalar = 0;

	uint32_t i;

	uint32_t sharevals[nvals];
	uint32_t sharevals_prime[nvals];
	uint32_t xvals[nvals];
	uint32_t yvals[nvals];

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
	for (i = 0; i < nvals; i++) {

		int32_t current_x = xembeddings[i];
		int32_t current_y = yembeddings[i];
		int32_t current_share = share_embeddings[i];
		int32_t current_share_prime = share_embeddings_prime[i];

		uint32_t *xptr = (uint32_t *)&current_x;
		uint32_t *yptr = (uint32_t *)&current_y;
		uint32_t *shareptr = (uint32_t *)&current_share;
		uint32_t *shareptr_prime = (uint32_t *)&current_share_prime;

		xvals[i] = *xptr;
		yvals[i] = *yptr;
		sharevals[i] = *shareptr;
		sharevals_prime[i] = *shareptr_prime;

		v_sum += xvals[i] * yvals[i];
	}
	uint32_t share_scalar_xvals;
	uint32_t share_scalar_yvals;
	float current_scalar_x = share_scalar_x[0];
	float current_scalar_y = share_scalar_y[0];

	uint32_t *scalar_xptr = (uint32_t *)&current_scalar_x;
	uint32_t *scalar_yptr = (uint32_t *)&current_scalar_y;

	share_scalar_xvals = *scalar_xptr;
	share_scalar_yvals = *scalar_yptr;
	// std::cout << "share_scalar_xvals " << share_scalar_xvals[0] << std::endl;
	// std::cout << "share_scalar_yvals " << share_scalar_xvals[10] << std::endl;
	//s_x_vec = ac->PutSIMDINGate(nvals, xvals.data(), 32, SERVER);
	//s_y_vec = ac->PutSIMDINGate(nvals, yvals.data(), 32, CLIENT);
	s_x_vec = ac->PutSharedSIMDINGate(nvals, sharevals_prime, bitlen);
	s_y_vec = ac->PutSharedSIMDINGate(nvals, sharevals, bitlen);
	/**
	 Step 7: Call the build method for building the circuit for the
	 problem by passing the shared objects and circuit object.
	 Don't forget to type cast the circuit object to type of share
	 */
	s_out = BuildInnerProductCircuit(s_x_vec, s_y_vec, nvals,
			(ArithmeticCircuit*) ac);

	/**
	 Step 8: Output the value of s_out (the computation result) to both parties
	 */
	s_out = ac->PutOUTGate(s_out, ALL);


	
	/**
	 Step 9: Executing the circuit using the ABYParty object evaluate the
	 problem.
	 */
	//party->ExecCircuit();


	// INPUTS
	share *magnitude_xin, *magnitude_yin;

	// // Input of the pre-computed shares of the face in the database
	magnitude_xin = bc->PutSharedINGate(share_scalar_xvals, bitlen);
	magnitude_yin = bc->PutSharedINGate(share_scalar_yvals, bitlen);
	
	share *s_x_times_y = bc->PutFPGate(magnitude_xin, magnitude_yin, MUL, bitlen, 1, no_status);

	s_out_scalar = bc->PutOUTGate(s_x_times_y, ALL);
	//s_out_scalar = bc->PutOUTGate(magnitude_xin, ALL);
	party->ExecCircuit();

	/**
	 Step 10: Type caste the plaintext output to 16 bit unsigned integer.
	 */
	output = s_out->get_clear_value<int32_t>();

	uint32_t *output_scalar_uint = (uint32_t *)s_out_scalar->get_clear_value_ptr();
	output_scalar = *((float *)output_scalar_uint);

	std::cout << std::endl << "cos_dist_ver: " << 1 - (v_sum / output_scalar) << std::endl;
	std::cout << "cos_dist: " << 1 - (output / output_scalar) << std::endl;

	// remove later
	std::cout << "output: " << output << std::endl;
	std::cout << "output_scalar: " << output_scalar << std::endl;
}

int main(int argc, char **argv)
{

	e_role role;
	uint32_t bitlen = 64, nvals = 128, secparam = 128, nthreads = 1;

	uint16_t port = 7766;
	std::string address = "127.0.0.1";
	int32_t test_op = -1;
	e_mt_gen_alg mt_alg = MT_OT;
	uint32_t test_bit = 0;
	uint32_t debug = 0;
	std::string inputfile;
	std::string pffrocd_path;

	read_test_options(&argc, &argv, &role, &bitlen, &nvals, &secparam, &address,
					  &port, &test_op, &test_bit, &mt_alg, &debug, &inputfile, &pffrocd_path);

	std::cout << std::fixed << std::setprecision(10);
	seclvl seclvl = get_sec_lvl(secparam);

	test_verilog_add64_SIMD(role, address, port, seclvl, nvals, nthreads, mt_alg, S_BOOL, debug, inputfile, pffrocd_path);

	return 0;
}

