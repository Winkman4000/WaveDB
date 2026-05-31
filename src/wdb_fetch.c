// WaveDB C string fetcher: random-access a dictionary value by code.
// Plain columns (mode 0): direct index. Front-coded (mode 1): jump to restart block,
// walk <=R deltas. Requires zstd to decompress the front-coded block once (cached).
// Usage: wdb_fetch <segment.wdb> <col> <code>            -> prints the value
//        wdb_fetch <segment.wdb> <col> bench <n>         -> times n random fetches
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
/* minimal zstd decl (no dev header needed; links against libzstd.so.1) */
extern size_t ZSTD_decompress(void* dst, size_t dstCap, const void* src, size_t srcSize);
extern unsigned ZSTD_isError(size_t code);
static double now_ms(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}

typedef struct {
    char name[128]; uint32_t V; uint8_t bits, dt, mode;
    // mode 0:
    const uint8_t* plain_dict;   // points at V*(u32 len|bytes)
    // mode 1:
    uint16_t R; uint32_t n_restart; const uint32_t* restarts;
    uint8_t* fc;                 // decompressed front-coded block (lazy)
    const uint8_t* fc_z; uint32_t fc_zlen, fc_len;
} Col;

static uint8_t* gbuf; static uint32_t N; static int NC; static Col cols[64];

static int load(const char* path){
    FILE*f=fopen(path,"rb"); if(!f)return -1; fseek(f,0,SEEK_END); long sz=ftell(f); rewind(f);
    gbuf=malloc(sz); if(fread(gbuf,1,sz,f)!=(size_t)sz)return -1; fclose(f);
    if(memcmp(gbuf,"WVDB3",5)){fprintf(stderr,"not WVDB3\n");return -1;}
    uint32_t off=5; NC=*(uint16_t*)(gbuf+off); off+=2; N=*(uint32_t*)(gbuf+off); off+=4;
    for(int c=0;c<NC;c++){
        uint16_t nl=*(uint16_t*)(gbuf+off); off+=2; memcpy(cols[c].name,gbuf+off,nl); cols[c].name[nl]=0; off+=nl;
        cols[c].V=*(uint32_t*)(gbuf+off); off+=4; cols[c].bits=gbuf[off++]; cols[c].dt=gbuf[off++]; cols[c].mode=gbuf[off++];
        cols[c].fc=NULL;
        if(cols[c].mode==0){
            cols[c].plain_dict=gbuf+off;
            for(uint32_t v=0;v<cols[c].V;v++){ uint32_t vl=*(uint32_t*)(gbuf+off); off+=4+vl; }
        } else {
            cols[c].R=*(uint16_t*)(gbuf+off); off+=2;
            cols[c].n_restart=*(uint32_t*)(gbuf+off); off+=4;
            cols[c].restarts=(const uint32_t*)(gbuf+off); off+=(uint64_t)cols[c].n_restart*4;
            cols[c].fc_len=*(uint32_t*)(gbuf+off); off+=4;
            cols[c].fc_zlen=*(uint32_t*)(gbuf+off); off+=4;
            cols[c].fc_z=gbuf+off; off+=cols[c].fc_zlen;
        }
        off+=(uint64_t)(N*cols[c].bits+7)/8;
    }
    return 0;
}
static int find(const char*n){for(int i=0;i<NC;i++)if(!strcmp(cols[i].name,n))return i;return -1;}

// ensure front-coded block decompressed
static void ensure_fc(Col*c){
    if(c->fc) return;
    c->fc=malloc(c->fc_len);
    size_t got=ZSTD_decompress(c->fc, c->fc_len, c->fc_z, c->fc_zlen);
    if(ZSTD_isError(got)){fprintf(stderr,"zstd err\n");exit(1);}
}
// fetch value for code -> writes into out (must be big enough), returns length
static int fetch(Col*c, uint32_t code, uint8_t* out){
    if(c->mode==0){
        const uint8_t* p=c->plain_dict;
        for(uint32_t v=0;v<code;v++){ uint32_t vl=*(uint32_t*)p; p+=4+vl; }
        uint32_t vl=*(uint32_t*)p; memcpy(out,p+4,vl); return vl;
    }
    ensure_fc(c);
    uint32_t blk=code/c->R; uint32_t o=c->restarts[blk]; int len=0;
    for(uint32_t j=0;j<=code % c->R;j++){
        uint16_t cp=*(uint16_t*)(c->fc+o); o+=2; uint16_t sl=*(uint16_t*)(c->fc+o); o+=2;
        // out[0..cp) keeps prev prefix; append suffix
        memcpy(out+cp, c->fc+o, sl); o+=sl; len=cp+sl;
    }
    return len;
}
int main(int argc,char**argv){
    if(argc<4){printf("usage: wdb_fetch <seg> <col> <code|bench n>\n");return 1;}
    if(load(argv[1])<0)return 1;
    int ci=find(argv[2]); if(ci<0){printf("no col %s\n",argv[2]);return 1;}
    Col*c=&cols[ci]; uint8_t* out=malloc(1<<20);
    if(!strcmp(argv[3],"bench")){
        int n=atoi(argv[4]); ensure_fc(c); srand(1);
        double t=now_ms(); volatile int s=0;
        for(int i=0;i<n;i++){ uint32_t code=rand()%c->V; s+=fetch(c,code,out); }
        double el=now_ms()-t;
        printf("%d fetches: %.1f ms total, %.3f µs/fetch (mode=%d R=%d)\n", n, el, el/n*1000, c->mode, c->mode?c->R:0);

    } else if(!strcmp(argv[3],"verify")){
        ensure_fc(c);
        // ground truth: sequential full decode of the whole front-coded block
        uint8_t* seq=malloc(1<<20); uint32_t o=0; int slen=0; long bad=0;
        uint8_t* got=malloc(1<<20);
        for(uint32_t code=0; code<c->V; code++){
            if(code % c->R==0) slen=0; // restart
            uint16_t cp=*(uint16_t*)(c->fc+o); o+=2; uint16_t sl=*(uint16_t*)(c->fc+o); o+=2;
            memcpy(seq+cp, c->fc+o, sl); o+=sl; slen=cp+sl;
            // independent random-access fetch
            int gl=fetch(c, code, got);
            if(gl!=slen || memcmp(got,seq,slen)){ bad++; if(bad<=3) fprintf(stderr,"mismatch code %u\n",code); }
        }
        printf("verify: %u/%u byte-exact (%ld mismatches)\n", c->V-(uint32_t)bad, c->V, bad);
        return bad?1:0;
    } else {
        uint32_t code=atoi(argv[3]); int len=fetch(c,code,out);
        fwrite(out,1,len,stdout); printf("\n");
    }
    return 0;
}
