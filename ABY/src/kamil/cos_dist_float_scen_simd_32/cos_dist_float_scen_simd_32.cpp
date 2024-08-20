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

void test_verilog_add64_SIMD(e_role role, const std::string &address, uint16_t port, seclvl seclvl, uint32_t nvals, uint32_t nthreads,
							 e_mt_gen_alg mt_alg, e_sharing sharing, uint32_t debug, std::string inputfile, std::string pffrocd_path)
{
	std::cout << "Start of test" << std::endl; 
	// std::cout << "SEC LEVEL: " << seclvl.symbits << std::endl;
	// std::cout << "MT_ALG: " << mt_alg << std::endl;

	// 32 bit version operating on floats
	uint32_t bitlen = 32;

	// two arrays of real-world embeddings
	std::vector<float> xembeddings;
	std::vector<float> yembeddings;

	// array for the Sy<role> share
	std::vector<float> share_embeddings;
	std::vector<float> share_embeddings_prime;


	// reading the non-xored embeddings, i.e. current face and database face

	std::fstream infile(inputfile);

	//std::cout << "INPUT FILE NAME: " << inputfile << std::endl;

	float x, y;

	//std::cout << "starting reading x and y" << std::endl;

	while (infile >> x >> y) {
		// std::cout << "x: " << x << " | y: "<< y << std::endl;
		xembeddings.push_back(x);
		yembeddings.push_back(y);
	}

	//std::cout<<"finished reading x and y" << std::endl;

	assert(xembeddings.size() == nvals);
	assert(yembeddings.size() == nvals);

	// reading the xored embedding, i.e. either Sy<0> or Sy<1> depending on the role

	// char *fname = (char *) malloc(150); // file name buffer 
    // sprintf(fname, "/home/dietpi/pffrocd/ABY/build/bin/share%d.txt", role);

	std::string fname = pffrocd_path + "/ABY/build/bin/share" + std::to_string(role) + ".txt";
	std::string fnameprime = pffrocd_path + "/ABY/build/bin/share" + std::to_string(role) + "prime.txt";
	//std::cout << "FNAME: " << fname << std::endl; 

	std::fstream infile_share(fname);
	std::fstream infile_share_prime(fnameprime);

	float z;
	// std::cout << "starting reading z" << std::endl;

	while(infile_share >> z) {
		//std::cout << "z: " << z << std::endl;
		share_embeddings.push_back(z);
	}
	z = 0;
	while(infile_share_prime >> z) {
		//std::cout << "z: " << z << std::endl;
		share_embeddings_prime.push_back(z);
	}

	//std::cout<<"finished reading z" << std::endl;

	assert(share_embeddings.size() == nvals);
	assert(share_embeddings_prime.size() == nvals);

	std::string circuit_dir = pffrocd_path + "/ABY/bin/circ/";

	std::cout << "CIRCUIT DIRECTORY: " << circuit_dir << std::endl;

	ABYParty *party = new ABYParty(role, address, port, seclvl, bitlen, nthreads, mt_alg, 100000, circuit_dir);

	std::cout << "party created" << std::endl;


	std::vector<Sharing *> &sharings = party->GetSharings();

	BooleanCircuit *bc = (BooleanCircuit *)sharings[S_BOOL]->GetCircuitBuildRoutine();
	ArithmeticCircuit *ac = (ArithmeticCircuit *)sharings[S_ARITH]->GetCircuitBuildRoutine();
	Circuit *yc = (BooleanCircuit *)sharings[S_YAO]->GetCircuitBuildRoutine();

	// std::cout << "circuit retrieved" << std::endl;
	// std::cout << "here 1" << std::endl;

	// arrays of integer pointers to doubles
	uint32_t xvals[nvals];
	uint32_t yvals[nvals];
	uint32_t sharevals[nvals];
	uint32_t sharevals_prime[nvals];
	// std::cout << "here 1" << std::endl;

	// verification in plaintext
	float ver_x_times_y[nvals];
	float ver_x_times_x[nvals];
	float ver_y_times_y[nvals];
	float ver_x_dot_y = 0;
	float ver_norm_x = 0;
	float ver_norm_y = 0;
	// std::cout << "here 1" << std::endl;

	// S_c(X,Y) = (X \dot Y) / (norm(X) * norm(Y))



	for (uint32_t i = 0; i < nvals; i++)
	{
		float current_x = xembeddings[i];
		float current_y = yembeddings[i];
		float current_share = share_embeddings[i];
		float current_share_prime = share_embeddings_prime[i];

		uint32_t *xptr = (uint32_t *)&current_x;
		uint32_t *yptr = (uint32_t *)&current_y;
		uint32_t *shareptr = (uint32_t *)&current_share;
		uint32_t *shareptr_prime = (uint32_t *)&current_share_prime;

		xvals[i] = *xptr;
		yvals[i] = *yptr;
		sharevals[i] = *shareptr;
		sharevals_prime[i] = *shareptr_prime;

		ver_x_times_y[i] = current_x * current_y;
		ver_x_dot_y += ver_x_times_y[i];

		ver_x_times_x[i] = current_x * current_x;
		ver_y_times_y[i] = current_y * current_y;
		ver_norm_x += ver_x_times_x[i];
		ver_norm_y += ver_y_times_y[i];
	}

	// std::cout << "here 1" << std::endl;
	//std::cout << "Do we reach this part of the program?" << std::endl;
	ver_norm_x = sqrt(ver_norm_x);
	ver_norm_y = sqrt(ver_norm_y);

	float ver_cos_sim = 1 - (ver_x_dot_y / (ver_norm_x * ver_norm_y));
	// INPUTS
	share *s_xin, *s_yin;

	// // Input of the pre-computed shares of the face in the database
	s_xin = bc->PutSharedSIMDINGate(nvals, sharevals_prime, bitlen);
	s_yin = bc->PutSharedSIMDINGate(nvals, sharevals, bitlen);

	share *s_x_times_y = bc->PutFPGate(s_xin, s_yin, MUL, bitlen, nvals, no_status);

	// computing x \dot y
	uint32_t posids[3] = {0, 0, 1};
	// share *s_product_first_wire = s_product->get_wire_ids_as_share(0);
	share *s_x_dot_y = bc->PutSubsetGate(s_x_times_y, posids, 1, true);
	for (int i = 1; i < nvals; i++)
	{
		//uint32_t posids[3] = {i, i, 1};

			posids[0] = i;
			posids[1] = i;
			posids[2] = 1;

		// share *s_product_split;
		s_x_dot_y = bc->PutFPGate(s_x_dot_y , bc->PutSubsetGate(s_x_times_y,posids,1,true),ADD);
	}

	share *x_dot_y_out = bc->PutOUTGate(s_x_dot_y, ALL);
	// std::cout << "finished computing cos dist" << std::endl;

	party->ExecCircuit();

	uint32_t out_bitlen_x_times_y, out_nvals_x_times_y;
	uint32_t *out_vals_x_times_y;


	std::cout << std::endl << "cos_dist_ver: " << ver_cos_sim << std::endl;

	uint32_t *x_dot_y_out_vals = (uint32_t *)x_dot_y_out->get_clear_value_ptr();
	float x_dot_y = *((float *)x_dot_y_out_vals);

	std::cout << "cos_dist: " << 1 - x_dot_y << std::endl;
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
