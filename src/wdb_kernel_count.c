// WaveDB C kernel v3: group-by backend for the integrated engine.
// Reads full.wdb (WVDB2). Materializes requested columns to byte/short codes ONCE (cached in memory
// across calls would need a daemon; here we do it per-invocation but only for the columns needed).
// Usage:
//   kernel2 gb <col1> [col2 col3]            -> single/multi-key GROUP BY count, prints top count + ms
//   kernel2 gbf <gcol> <fcol> <op> <val>     -> filtered GROUP BY count (op: eq/ne/gt/lt), top count + ms
// Reads codes; for grouping we only need codes (not decoded values) -> fast.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
static double now_ms(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
typedef struct { char name[128]; uint32_t V; uint8_t bits; uint8_t dt; uint8_t* packed; long long* ivals; uint32_t* codes; } Col;
static uint32_t N; static int NC; static Col cols[64]; static uint8_t* gbuf;
static int load(const char* path){
    FILE*f=fopen(path,"rb"); if(!f)return -1; fseek(f,0,SEEK_END); long sz=ftell(f); rewind(f);
    gbuf=malloc(sz); if(fread(gbuf,1,sz,f)!=(size_t)sz)return -1; fclose(f);
    uint32_t off=5; NC=*(uint16_t*)(gbuf+off); off+=2; N=*(uint32_t*)(gbuf+off); off+=4;
    for(int c=0;c<NC;c++){ uint16_t nl=*(uint16_t*)(gbuf+off); off+=2; memcpy(cols[c].name,gbuf+off,nl); cols[c].name[nl]=0; off+=nl;
        cols[c].V=*(uint32_t*)(gbuf+off); off+=4; cols[c].bits=gbuf[off]; off+=1; cols[c].dt=gbuf[off]; off+=1;
        uint8_t mode=gbuf[off]; off+=1;
        cols[c].ivals=malloc(sizeof(long long)*cols[c].V);
        if(mode==0){
            for(uint32_t v=0;v<cols[c].V;v++){ uint32_t vl=*(uint32_t*)(gbuf+off); off+=4; char tmp[32]; int n=vl<31?vl:31; memcpy(tmp,gbuf+off,n);tmp[n]=0; cols[c].ivals[v]=(cols[c].dt==0)?atoll(tmp):0; off+=vl; }
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
#define NT 8
// parallel multi-key group-by count over an optional mask. Keys combined into one combo via radix.
typedef struct{uint32_t lo,hi;const uint32_t**kc;int nk;const uint32_t*Vs;const uint8_t*mask;uint64_t*ht;uint64_t htsize;uint64_t*top;}Job;
static void* w(void*a){Job*j=(Job*)a;uint64_t*ht=calloc(j->htsize,8);
    for(uint32_t i=j->lo;i<j->hi;i++){ if(j->mask&&!j->mask[i])continue; uint64_t key=0; for(int k=0;k<j->nk;k++) key=key*j->Vs[k]+j->kc[k][i]; ht[key]++; }
    // reduce into shared via local max only (we just need the global top count): merge by returning local ht through top
    // To get exact global top we must merge full ht; store pointer
    j->top[0]=(uint64_t)ht; return NULL;}
static uint64_t groupby(int*gci,int nk,const uint8_t*mask,double*ms){
    const uint32_t* kc[3]; uint32_t Vs[3]; uint64_t htsize=1;
    for(int k=0;k<nk;k++){materialize(&cols[gci[k]]);kc[k]=cols[gci[k]].codes;Vs[k]=cols[gci[k]].V;htsize*=cols[gci[k]].V;}
    double t0=now_ms();
    // if combined keyspace small enough, dense histogram per thread; else fall to single-thread dense (still ok)
    pthread_t th[NT];Job jobs[NT];uint64_t tops[NT];uint32_t per=(N+NT-1)/NT;
    for(int t=0;t<NT;t++){jobs[t]=(Job){t*per,(uint32_t)((t+1)*per>N?N:(t+1)*per),kc,nk,Vs,mask,NULL,htsize,&tops[t]};pthread_create(&th[t],NULL,w,&jobs[t]);}
    for(int t=0;t<NT;t++)pthread_join(th[t],NULL);
    // merge the NT partial histograms, find global top
    uint64_t top=0; uint64_t* m=calloc(htsize,8);
    for(int t=0;t<NT;t++){uint64_t*h=(uint64_t*)tops[t];for(uint64_t k=0;k<htsize;k++)m[k]+=h[k];free(h);}
    for(uint64_t k=0;k<htsize;k++) if(m[k]>top)top=m[k];
    free(m); *ms=now_ms()-t0; return top;
}
int main(int argc,char**argv){
    if(argc<3){printf("usage: wdb_kernel_count <segment.wdb> gb <col...>\n");return 1;}
    if(load(argv[1])<0){printf("loadfail\n");return 1;}
    double mt0=now_ms();
    if(argc>=4 && !strcmp(argv[2],"gb")){
        int nk=argc-3; int gci[3]; for(int k=0;k<nk;k++){gci[k]=find(argv[3+k]); if(gci[k]<0){printf("ERR nocol %s\n",argv[2+k]);return 1;}}
        double ms; uint64_t top=groupby(gci,nk,NULL,&ms);
        printf("top=%llu ms=%.3f\n",(unsigned long long)top,ms);
    } else if(argc>=7 && !strcmp(argv[2],"gbf")){
        int gci=find(argv[3]); int fci=find(argv[4]); if(gci<0||fci<0){printf("ERR\n");return 1;}
        materialize(&cols[fci]); long long val=atoll(argv[6]); const char*op=argv[5];
        uint8_t* mask=malloc(N);
        Col*fc=&cols[fci];
        for(uint32_t i=0;i<N;i++){ long long v=fc->ivals[fc->codes[i]]; int keep=0;
            if(!strcmp(op,"eq"))keep=(v==val); else if(!strcmp(op,"ne"))keep=(v!=val); else if(!strcmp(op,"gt"))keep=(v>val); else if(!strcmp(op,"lt"))keep=(v<val);
            mask[i]=keep; }
        double ms; uint64_t top=groupby(&gci,1,mask,&ms);
        printf("top=%llu ms=%.3f\n",(unsigned long long)top,ms);
    }
    (void)mt0; return 0;
}
