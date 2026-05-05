#include "device/index.h"
Device device{"v6.3.6"};


void setup(){
    Serial.begin(115200);
    delay(700);

    // device.sensors.pressure.debug = false;
    device.setup();
}

void loop(){
    device.tasks.handle();
}
