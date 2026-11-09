#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do { CUresult r=(x); if(r!=CUDA_SUCCESS){const char* n=nullptr;cuGetErrorName(r,&n);std::fprintf(stderr,"%s: %s\n",#x,n?n:"unknown");std::exit(1);} } while(0)
int main(){
 CHECK(cuInit(0)); CUdevice dev;CHECK(cuDeviceGet(&dev,0));CUcontext ctx;CHECK(cuDevicePrimaryCtxRetain(&ctx,dev));CHECK(cuCtxSetCurrent(ctx));
 CUstream main,alt;CHECK(cuStreamCreate(&main,CU_STREAM_NON_BLOCKING));CHECK(cuStreamCreate(&alt,CU_STREAM_NON_BLOCKING));
 CUdeviceptr shared,routed;CHECK(cuMemAlloc(&shared,1024));CHECK(cuMemAlloc(&routed,1024));void* host;CHECK(cuMemAllocHost(&host,1024));
 CUgraph child;CHECK(cuGraphCreate(&child,0));CUDA_MEMSET_NODE_PARAMS p={};p.dst=shared;p.pitch=0;p.value=3;p.elementSize=4;p.width=256;p.height=1;CUgraphNode child_memset;CHECK(cuGraphAddMemsetNode(&child_memset,child,nullptr,0,&p,ctx));
 CUevent fork,join;CHECK(cuEventCreate(&fork,CU_EVENT_DISABLE_TIMING));CHECK(cuEventCreate(&join,CU_EVENT_DISABLE_TIMING));
 CHECK(cuStreamBeginCapture(main,CU_STREAM_CAPTURE_MODE_RELAXED));CHECK(cuEventRecord(fork,main));CHECK(cuStreamWaitEvent(alt,fork,0));
 CUstreamCaptureStatus status;cuuint64_t id;CUgraph parent;const CUgraphNode* deps;size_t count;
 CHECK(cuStreamGetCaptureInfo_v3(alt,&status,&id,&parent,&deps,nullptr,&count));if(status!=CU_STREAM_CAPTURE_STATUS_ACTIVE) return 2;
 std::vector<CUgraphNode> copy;if(count)copy.assign(deps,deps+count);CUgraphNode inserted;CHECK(cuGraphAddChildGraphNode(&inserted,parent,copy.empty()?nullptr:copy.data(),copy.size(),child));
 CHECK(cuStreamUpdateCaptureDependencies_v2(alt,&inserted,nullptr,1,CU_STREAM_SET_CAPTURE_DEPENDENCIES));
 CHECK(cuMemsetD32Async(routed,2,256,main));CHECK(cuEventRecord(join,alt));CHECK(cuStreamWaitEvent(main,join,0));CHECK(cuMemcpyDtoHAsync(host,shared,1024,main));
 CUgraph captured;CHECK(cuStreamEndCapture(main,&captured));CUgraphExec exec;CHECK(cuGraphInstantiateWithFlags(&exec,captured,0));CHECK(cuGraphLaunch(exec,main));CHECK(cuStreamSynchronize(main));
 for(int i=0;i<256;++i)if(static_cast<unsigned*>(host)[i]!=3)return 3;
 std::puts("child capture dependency insertion and join passed");CHECK(cuGraphExecDestroy(exec));CHECK(cuGraphDestroy(captured));CHECK(cuGraphDestroy(child));CHECK(cuEventDestroy(join));CHECK(cuEventDestroy(fork));CHECK(cuMemFreeHost(host));CHECK(cuMemFree(routed));CHECK(cuMemFree(shared));CHECK(cuStreamDestroy(alt));CHECK(cuStreamDestroy(main));CHECK(cuDevicePrimaryCtxRelease(dev));
}
