#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/grpcpp.h>
#include <grpc/grpc.h>
#include "NicSysHisRpc.pb.h"
#include "NicSysHisRpc.grpc.pb.h"



#if defined DEBUG || defined _DEBUG
#pragma comment(lib, "ws2_32.lib") 
#pragma comment(lib,"libprotobufd.lib")
#pragma comment(lib,"libprotocd.lib")
#pragma comment(lib,"gpr.lib")
#pragma comment(lib,"grpc.lib")
#pragma comment(lib,"grpc_cronet.lib")
#pragma comment(lib,"grpc_unsecure.lib")
#pragma comment(lib,"grpc_plugin_support.lib")
#pragma comment(lib,"grpcpp_channelz.lib")
#pragma comment(lib,"cares.lib")
#pragma comment(lib,"address_sorting.lib")
#pragma comment(lib,"grpc++.lib")
#pragma comment(lib,"grpc++_cronet.lib")
#pragma comment(lib,"grpc++_unsecure.lib")
#pragma comment(lib,"grpc++_error_details.lib")
#pragma comment(lib,"grpc++_reflection.lib")
#pragma comment(lib,"grpcpp_channelz.lib")

#pragma comment(lib,"zlibd.lib")
#else
#pragma comment(lib,"libprotobuf.lib")
#pragma comment(lib,"libprotoc.lib")
#pragma comment(lib,"gpr.lib")
#pragma comment(lib,"grpc.lib")
#pragma comment(lib,"grpc++.lib")
#endif // DEBUG

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using NicSys_Hisd::HisdService;
using NicSys_Hisd::psHisWriteParameter;
using NicSys_Hisd::psHisRPCReturn;
using NicSys_Hisd::psHisQueryParameter;
using NicSys_Hisd::psDataField;
using NicSys_Hisd::psTagHisData;
using NicSys_Hisd::psData;
using NicSys_Hisd::psTime;
using NicSys_Hisd::psVariant;
using NicSys_Hisd::QualityInfo;
using NicSys_Hisd::psDataTypeEnum;


class HisdServiceClient {
public:
	HisdServiceClient(std::shared_ptr<Channel> channel)
		: stub_(HisdService::NewStub(channel)) {}

	//测试写入和查询接口
	int Trends_Write_And_Query() {
		// Data we are sending to the server.
		psDataField *field = new psDataField();
		field->set_quality(1);
		field->set_time(10102);
		field->set_value(34);

		psHisWriteParameter  request;
		request.set_allocated_fields(field);

	
		//设置5个id
		for (int id = 1; id < 5; id++)
		{
			psTagHisData *td = request.add_ptaghisdata();
			td->set_tagid(id);

			//每个id下带10个points
			for (int k = 0; k < 10; k++)
			{
				psData * data = td->add_datalist();
				psTime *time = new psTime();
				psVariant * val = new psVariant();
				QualityInfo quality;

				time->set_second(10011 + k);
				time->set_millisec(21);
				val->set_datatype(psDataTypeEnum::psDataType_Int32);
				val->set_int32(6788 + k);
				quality.set_q(1);

				data->set_allocated_time(time);
				data->set_quality(quality.q());
				data->set_allocated_value(val);
			}
		}
		
		
		psHisRPCReturn reply;

		ClientContext context;
		Status status = stub_->His_Write(&context, request, &reply);

		request.clear_ptaghisdata();

		if (status.ok()) {

			return reply.apistatus();
		}
		else {
			std::cout << status.error_code() << ": " << status.error_message()
				<< std::endl;
			return -1;
		}
	}

private:
	std::unique_ptr<HisdService::Stub> stub_;
};

int main(int argc, char** argv) {
	// Instantiate the client. It requires a channel, out of which the actual RPCs
	// are created. This channel models a connection to an endpoint (in this case,
	// localhost at port 50051). We indicate that the channel isn't authenticated
	// (use of InsecureChannelCredentials()).
	HisdServiceClient client(grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials()));
	int reply = client.Trends_Write_And_Query();
	std::cout << "Hisd received: " << reply << std::endl;
	system("PAUSE");
	return 0;
}