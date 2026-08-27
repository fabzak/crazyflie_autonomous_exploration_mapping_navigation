#include "datum_lifecycle.h"
#include <stdio.h>
#include <string.h>
int main(void){
    dl_t dl; dl_output_t o; dl_input_t in; uint32_t t=0;
    dl_init(&dl); memset(&in,0,sizeof(in));
    in.now_ms=t; in.clearance_valid=true; in.clearance_m=0.40f;
    in.phase=DL_PHASE_HOLD_LAYER; in.phase_age_ms=0u;
    dl_update(&dl,&in,&o); t+=25u;
    for(int i=0;i<24;i++){
        in.now_ms=t; in.phase_age_ms=1000u;
        dl_update(&dl,&in,&o); t+=25u;
        if(i>=17) printf("  %6d %9s  %-18s %9s %11s\n", i, "1.0s",
            dl_state_name(o.state), o.detector_enabled?"True":"False",
            o.xy_allowed?"True":"False");
    }
    return 0;
}
