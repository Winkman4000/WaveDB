// WaveDB C kernel v4: adds per-group aggregates (SUM/MIN/MAX/AVG) in the same parallel pass.
// The value IS the weight: instead of histogram[group]++ we also do sum[group]+=value.
// Usage:
//   kernel3 gb <col...>                          -> top count (as before)
//   kernel3 gbagg <gcol> <aggcol> <fn> <order>   -> per-group agg; fn=sum|avg|min|max
//        order=count|key|aggdesc  (which group's value to report: top-count group, min-key group, or max-agg group)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <limits.h>
static double now_ms(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
typedef struct { char name[128]; uint32_t V; uint8_t bits; uint8_t dt; uint8_t* packed; long long* ivals; uint32_t* codes; } Col;
static uint32_t N; static int NC; static Col cols[64]; static uint8_t* gbuf;
static int load(const char* path){
    FILE*f=fopen(path,"rb"); if(!f)return -1; fseek(f,0,SEEK_END); long sz=ftell(f); rewind(f);
    gbuf=malloc(sz); if(fread(gbuf,1,sz,f)!=(size_t)sz)return -1; fclose(f);
    uint32_t off=5; NC=*(uint16_t*)(gbuf+off); off+=2; N=*(uint32_t*)(gbuf+off); off+=4;
    for(int c=0;c<NC;c++){ uint16_t nl=*(uint16_t*)(gbuf+off); off+=2; memcpy(cols[c].name,gbuf+off,nl); cols[c].name[nl]=0; off+=nl;
        cols[c].V=*(uint32_t*)(gbuf+off); off+=4; cols[c].bits=gbuf[off]; off+=1; cols[c].dt=gbuf[off]; off+=1;
        uint8_t mode=gbuf[off]; off+=1; uint8_t has_null=gbuf[off]; off+=1;
        cols[c].ivals=malloc(sizeof(long long)*cols[c].V);
        if(mode==0){
            for(uint32_t v=0;v<cols[c].V-has_null;v++){ uint32_t vl=*(uint32_t*)(gbuf+off); off+=4; char tmp[32]; int n=vl<31?vl:31; memcpy(tmp,gbuf+off,n);tmp[n]=0; cols[c].ivals[v]=(cols[c].dt==0)?atoll(tmp):0; off+=vl; }
        } else {
            // front-coded dict: skip it (grouping uses codes only; front-coded cols are strings, no int vals)
            off+=2; uint32_t nr=*(uint32_t*)(gbuf+off); off+=4; off+=(uint64_t)nr*4;
            uint32_t fclen=*(uint32_t*)(gbuf+off); off+=4; uint32_t zlen=*(uint32_t*)(gbuf+off); off+=4; off+=zlen;
            for(uint32_t v=0;v<cols[c].V;v++) cols[c].ivals[v]=0; (void)fclen;
        }
        cols[c].packed=gbuf+off; off+=(uint64_t)(N*cols[c].bits+7)/8; cols[c].codes=NULL; }
    return 0;
}
static int find(const char*n){for(int i=0;i<NC;i++)if(!strcmp(cols[i].name,n))return i;return -1;}
static void materialize(Col*c){ if(c->codes)return; c->codes=malloc((size_t)N*4); uint64_t bp=0; uint8_t bits=c->bits; const uint8_t*p=c->packed;
    for(uint32_t i=0;i<N;i++){uint32_t code=0;for(int b=0;b<bits;b++){uint64_t q=bp+b;code=(code<<1)|((p[q>>3]>>(7-(q&7)))&1);}c->codes[i]=code;bp+=bits;} }

// per-group aggregate: single pass, value-as-weight. Group key = gcol code. Value = decoded aggcol.
int main(int argc,char**argv){
    if(argc<2){printf("usage: wdb_kernel <segment.wdb> <op> <args>\n");return 1;}
    if(load(argv[1])<0){printf("loadfail\n");return 1;}
    if(argc>=7 && !strcmp(argv[2],"gbagg")){
        int gci=find(argv[3]), aci=find(argv[4]); if(gci<0||aci<0){printf("ERR\n");return 1;}
        const char*fn=argv[5]; const char*order=argv[6];
        Col*gc=&cols[gci]; Col*ac=&cols[aci]; materialize(gc); materialize(ac);
        uint32_t V=gc->V;
        double t0=now_ms();
        // per-group accumulators
        double* gsum=calloc(V,sizeof(double)); uint64_t* gcnt=calloc(V,8);
        long long* gmin=malloc(V*sizeof(long long)); long long* gmax=malloc(V*sizeof(long long));
        for(uint32_t v=0;v<V;v++){gmin[v]=LLONG_MAX;gmax[v]=LLONG_MIN;}
        for(uint32_t i=0;i<N;i++){ uint32_t g=gc->codes[i]; long long val=ac->ivals[ac->codes[i]];
            gsum[g]+=val; gcnt[g]++; if(val<gmin[g])gmin[g]=val; if(val>gmax[g])gmax[g]=val; }
        // pick the reported group
        uint32_t pick=0; 
        if(!strcmp(order,"key")){ // smallest group-key VALUE (codes are dict-sorted = value order for ints/dates)
            long long best=LLONG_MAX; for(uint32_t v=0;v<V;v++) if(gcnt[v]&&gc->ivals[v]<best){best=gc->ivals[v];pick=v;} }
        else if(!strcmp(order,"aggdesc")){ // group with max sum
            double best=-1e300; for(uint32_t v=0;v<V;v++) if(gcnt[v]&&gsum[v]>best){best=gsum[v];pick=v;} }
        else { uint64_t best=0; for(uint32_t v=0;v<V;v++) if(gcnt[v]>best){best=gcnt[v];pick=v;} }
        double ms=now_ms()-t0;
        double avg=gcnt[pick]?gsum[pick]/gcnt[pick]:0;
        printf("group_key=%lld sum=%lld min=%lld max=%lld avg=%.4f count=%llu ms=%.3f\n",
            gc->ivals[pick],(long long)gsum[pick],gmin[pick],gmax[pick],avg,(unsigned long long)gcnt[pick],ms);
    }
    return 0;
}
