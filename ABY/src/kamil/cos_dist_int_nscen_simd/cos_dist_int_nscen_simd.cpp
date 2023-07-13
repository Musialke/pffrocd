
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
					   uint16_t *port, int32_t *test_op, uint32_t *test_bit, double *fpa, double *fpb, std::string *inputfile, std::string *outputfile)
{

	uint32_t int_role = 0, int_port = 0, int_testbit = 0;

	parsing_ctx options[] =
		{{(void *)&int_role, T_NUM, "r", "Role: 0/1", true, false},
		 {(void *)&int_testbit, T_NUM, "i", "test bit", false, false},
		 {(void *)nvals, T_NUM, "n", "Number of parallel operation elements", false, false},
		 {(void *)bitlen, T_NUM, "b", "Bit-length, default 32", false, false},
		 {(void *)secparam, T_NUM, "s", "Symmetric Security Bits, default: 128", false, false},
		 {(void *)address, T_STR, "a", "IP-address, default: localhost", false, false},
		 {(void *)&int_port, T_NUM, "p", "Port, default: 7766", false, false},
		 {(void *)test_op, T_NUM, "t", "Single test (leave out for all operations), default: off", false, false},
		 {(void *)fpa, T_DOUBLE, "x", "FP a", false, false},
		 {(void *)fpb, T_DOUBLE, "y", "FP b", false, false},
		 {(void *)inputfile, T_STR, "f", "Input file containing face embeddings", true, false},
		 {(void *)outputfile, T_STR, "o", "Output file containing cos sim result", false, false}
		};

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
}

void test_verilog_add64_SIMD(e_role role, const std::string &address, uint16_t port, seclvl seclvl, uint32_t nvals, uint32_t nthreads,
							 e_mt_gen_alg mt_alg, e_sharing sharing, double afp, double bfp, std::string inputfile, std::string outputfile)
{

	// set bitlen to 16 like in the innerproduct example
	uint32_t bitlen = 32;

	// two arrays of real-world embeddings
	std::vector<uint32_t> xembeddings;
	std::vector<uint32_t> yembeddings;

	// reading the non-xored embeddings, i.e. current face and database face

	std::fstream infile(inputfile);

	uint32_t x, y;

	while (infile >> x >> y) {
		//std::cout << "x: " << x << " | y: "<< y << std::endl;
		xembeddings.push_back(x);
		yembeddings.push_back(y);
	}

	assert(xembeddings.size() == nvals);
	assert(yembeddings.size() == nvals);

	std::string circuit_dir = "../../bin/circ/";

	ABYParty *party = new ABYParty(role, address, port, seclvl, bitlen, nthreads, mt_alg, 100000, circuit_dir);

	std::vector<Sharing *> &sharings = party->GetSharings();

	BooleanCircuit *bc = (BooleanCircuit *)sharings[S_BOOL]->GetCircuitBuildRoutine();

	// verification in plaintext
	double ver_x_times_y[nvals];
	double ver_x_times_x[nvals];
	double ver_y_times_y[nvals];
	double ver_x_dot_y = 0;
	double ver_norm_x = 0;
	double ver_norm_y = 0;

	// S_c(X,Y) = (X \dot Y) / (norm(X) * norm(Y))

	for (uint32_t i = 0; i < nvals; i++)
	{
		double current_x = xembeddings[i];
		double current_y = yembeddings[i];

		ver_x_times_y[i] = current_x * current_y;
		ver_x_dot_y += ver_x_times_y[i];

		ver_x_times_x[i] = current_x * current_x;
		ver_y_times_y[i] = current_y * current_y;
		ver_norm_x += ver_x_times_x[i];
		ver_norm_y += ver_y_times_y[i];

	}

	ver_norm_x = sqrt(ver_norm_x);
	ver_norm_y = sqrt(ver_norm_y);

	double ver_cos_sim = 1 - (ver_x_dot_y / (ver_norm_x * ver_norm_y));

	// INPUTS
	share *s_xin = bc->PutSIMDINGate(nvals, xembeddings.data(), bitlen, SERVER);
	share *s_yin = bc->PutSIMDINGate(nvals, yembeddings.data(), bitlen, CLIENT);

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

		// bc->PutPrintValueGate(bc->PutSubsetGate(s_product,posids,1,false), "First wire");

		// share *s_product_split;
		s_x_dot_y = bc->PutFPGate(s_x_dot_y , bc->PutSubsetGate(s_x_times_y,posids,1,true),ADD);
		//std::cout << "s_share nvals: " << a_share->get_nvals() << std::endl;
		//std::cout << "s_share bitlen: " << a_share->get_bitlength() << std::endl;
		//bc->PutPrintValueGate(s_x_dot_y, "s_x_dot_y");
	}
	//bc->PutPrintValueGate(s_x_dot_y, "s_x_dot_y");
	share *s_x_dot_y_out = bc->PutOUTGate(s_x_dot_y, ALL);


	// computing norm(X)

	share *s_x_times_x = bc->PutFPGate(s_xin, s_xin, MUL, bitlen, nvals, no_status);
	// bc->PutPrintValueGate(s_x_times_x, "s_x_times_x");


	posids[0] = 0;
	posids[1] = 0;
	posids[2] = 1;
	// share *s_product_first_wire = s_product->get_wire_ids_as_share(0);
	share *s_norm_x = bc->PutSubsetGate(s_x_times_x, posids, 1, true);
	for (int i = 1; i < nvals; i++)
	{
		//uint32_t posids[3] = {i, i, 1};

			posids[0] = i;
			posids[1] = i;
			posids[2] = 1;

		// bc->PutPrintValueGate(bc->PutSubsetGate(s_product,posids,1,false), "First wire");

		// share *s_product_split;
		s_norm_x = bc->PutFPGate(s_norm_x , bc->PutSubsetGate(s_x_times_x,posids,1,true),ADD);
		//std::cout << "s_share nvals: " << a_share->get_nvals() << std::endl;
		//std::cout << "s_share bitlen: " << a_share->get_bitlength() << std::endl;
		//bc->PutPrintValueGate(a_share, "a_share");
	}

	s_norm_x = bc->PutFPGate(s_norm_x, SQRT);
	// bc->PutPrintValueGate(s_norm_x, "s_norm_x");

	share *s_norm_x_out = bc->PutOUTGate(s_norm_x, ALL);


	// computing norm(Y)

	share *s_y_times_y = bc->PutFPGate(s_yin, s_yin, MUL, bitlen, nvals, no_status);
	// bc->PutPrintValueGate(s_y_times_y, "s_y_times_y");


	posids[0] = 0;
	posids[1] = 0;
	posids[2] = 1;
	// share *s_product_first_wire = s_product->get_wire_ids_as_share(0);
	share *s_norm_y = bc->PutSubsetGate(s_y_times_y, posids, 1, true);
	for (int i = 1; i < nvals; i++)
	{
		//uint32_t posids[3] = {i, i, 1};

			posids[0] = i;
			posids[1] = i;
			posids[2] = 1;

		// bc->PutPrintValueGate(bc->PutSubsetGate(s_product,posids,1,false), "First wire");

		// share *s_product_split;
		s_norm_y = bc->PutFPGate(s_norm_y , bc->PutSubsetGate(s_y_times_y,posids,1,true),ADD);
		//std::cout << "s_share nvals: " << a_share->get_nvals() << std::endl;
		//std::cout << "s_share bitlen: " << a_share->get_bitlength() << std::endl;
		//bc->PutPrintValueGate(a_share, "a_share");
	}

	s_norm_y = bc->PutFPGate(s_norm_y, SQRT);
	// bc->PutPrintValueGate(s_norm_y, "s_norm_y");

	share *s_norm_y_out = bc->PutOUTGate(s_norm_y, ALL);


	share *s_norm_x_times_norm_y = bc->PutFPGate(s_norm_x, s_norm_y, MUL);
	// bc->PutPrintValueGate(s_norm_x_times_norm_y, "s_norm_x_times_norm_y");

	share *s_cos_sim = bc->PutFPGate(s_x_dot_y, s_norm_x_times_norm_y, DIV);
	// bc->PutPrintValueGate(s_cos_sim, "s_cos_sim");

	share *s_cos_sim_out = bc->PutOUTGate(s_cos_sim, ALL);


	party->ExecCircuit();

	// printing

	std::cout << "INPUT EMBEDDINGS:" << std::endl;

	std::cout << "X:" << std::endl;

	for (int i = 0; i < nvals; i++) {
		std::cout << xembeddings[i] << ", ";
	}

	std::cout << std::endl;

	std::cout << "Y:" << std::endl;

	for (int i = 0; i < nvals; i++) {
		std::cout << yembeddings[i] << ", ";
	}


	std::cout << std::endl;

	std::cout << "VERIFICATION:" << std::endl;
	
	std::cout << "x dot y: " << ver_x_dot_y << std::endl;

	std::cout << "norm(x): " << ver_norm_x << std::endl;

	std::cout << "norm(y): " << ver_norm_y << std::endl;

	std::cout << "cos sim: " << ver_cos_sim << std::endl;

	std::cout << "CIRCUIT RESULTS:" << std::endl;

	uint32_t cos_sim = s_cos_sim_out->get_clear_value<uint32_t>();

	uint32_t x_dot_y = s_x_dot_y_out->get_clear_value<uint32_t>();

	uint32_t norm_x = s_norm_x_out->get_clear_value<uint32_t>();

	uint32_t norm_y = s_norm_y_out->get_clear_value<uint32_t>();

	std::cout << "x dot share: " << x_dot_y << std::endl;
	std::cout << "norm(x) : " << norm_x << std::endl;
	std::cout << "norm(share): " << norm_y << std::endl;
	std::cout << "cos sim: " << 1 - cos_sim << std::endl;	
}

int main(int argc, char **argv)
{

	e_role role;
	// set bitlen to 32
	uint32_t bitlen = 32, nvals = 128, secparam = 128, nthreads = 1;

	uint16_t port = 7766;
	std::string address = "127.0.0.1";
	int32_t test_op = -1;
	e_mt_gen_alg mt_alg = MT_OT;
	uint32_t test_bit = 0;
	double fpa = 10.52, fpb = 1.30;
	std::string inputfile;
	std::string outputfile;

	read_test_options(&argc, &argv, &role, &bitlen, &nvals, &secparam, &address,
					  &port, &test_op, &test_bit, &fpa, &fpb, &inputfile, &outputfile);

	std::cout << std::fixed << std::setprecision(10);
	seclvl seclvl = get_sec_lvl(secparam);

	test_verilog_add64_SIMD(role, address, port, seclvl, nvals, nthreads, mt_alg, S_BOOL, fpa, fpb, inputfile, outputfile);

	return 0;
}
