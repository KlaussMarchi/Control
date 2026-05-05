import json
import os
import sys

class ArduinoCodeGenerator:
    def __init__(self, infoPath, controllerInfoPath):
        with open(infoPath, 'r') as f:
            self.info = json.load(f)

        with open(controllerInfoPath, 'r') as f:
            self.ctlInfo = json.load(f)

    def generateDefines(self):
        dt = self.info['dt']
        offset = self.info['offset']
        sensorRange = self.info['sensor_range']
        actuatorRange = self.info['actuator_range']

        initX = round(sum(sensorRange) / 2, 4)
        initU = round(sum(actuatorRange) / 2, 1)

        return f"""#define DT         {dt}
#define OFFSET     {offset}
#define TS_FILTER  15.0
#define SENSOR_MIN {sensorRange[0]}
#define SENSOR_MAX {sensorRange[1]}

#define MIN_OUT {actuatorRange[0]}
#define MAX_OUT {actuatorRange[1]}

#define INIT_X  {initX}
#define INIT_U  {initU}"""

    def generateLowPassFilter(self):
        num = self.info['sensor_filter']['num']
        den = self.info['sensor_filter']['den']
        order = len(num)

        return f"""class LowPassFilter {{
  private:
    float num[{order}];
    float den[{order}];
    float Xn[{order}];
    float Yn[{order}];

  public:
    LowPassFilter(float n0, float n1, float d0, float d1) {{
      num[0] = n0; num[1] = n1;
      den[0] = d0; den[1] = d1;
      for (int i = 0; i < {order}; i++) {{
        Xn[i] = 0.0;
        Yn[i] = 0.0;
      }}
    }}

    void reset(float value) {{
      for (int i = 0; i < {order}; i++) {{
        Xn[i] = value;
        Yn[i] = value;
      }}
    }}

    float update(float inputValue) {{
      for (int n = {order - 1}; n > 0; n--) {{
        Xn[n] = Xn[n - 1];
        Yn[n] = Yn[n - 1];
      }}
      Xn[0] = inputValue;

      float out = 0.0;
      for (int i = 0; i < {order}; i++)
        out += Xn[i] * num[i];
      for (int i = 1; i < {order}; i++)
        out -= Yn[i] * den[i];

      Yn[0] = out;
      return out;
    }}
}};"""

    def generateController(self):
        num = self.ctlInfo['num']
        den = self.ctlInfo['den']
        order = len(num)
        
        return f"""class Controller {{
  private:
    float num[{order}];
    float den[{order}];
    float Xn[{order}];
    float Yn[{order}];

  public:
    Controller(float initX, float initU) {{
      num[0] = {num[0]}; num[1] = {num[1]}; num[2] = {num[2]};
      den[0] = {den[0]}; den[1] = {den[1]}; den[2] = {den[2]};
      
      for (int i = 0; i < {order}; i++) {{
        Xn[i] = 0.0;
        Yn[i] = initU;
      }}
    }}

    float update(float setpoint, float sensorValue) {{
        float error = setpoint - sensorValue;
        
        for (int n = {order - 1}; n > 0; n--) {{
            Xn[n] = Xn[n - 1];
            Yn[n] = Yn[n - 1];
        }}
        Xn[0] = error;

        float out = 0.0;
        for (int i = 0; i < {order}; i++)
            out += Xn[i] * num[i];
            
        for (int i = 1; i < {order}; i++)
            out -= Yn[i] * den[i];

        if (out < MIN_OUT) out = MIN_OUT;
        if (out > MAX_OUT) out = MAX_OUT;

        Yn[0] = out;
        return out;
    }}
}};"""

    def generateSetpointer(self):
        return """class Setpointer {
  private:
    float ref;
    float alpha;
    float target;

  public:
    Setpointer(float initRef, float settlingTime, float sampleTime)
        : ref(initRef), target(initRef) {
        alpha = exp(-4.0 * (sampleTime / settlingTime));
    }

    void update(float newTarget) {
        target = newTarget;
    }

    float tick() {
        ref = alpha * ref + (1.0 - alpha) * target;
        return ref;
    }

    float getTarget() { return target; }
    float getRef()    { return ref; }
};"""

    def generateFull(self):
        num = self.info['sensor_filter']['num']
        den = self.info['sensor_filter']['den']

        filterInit = f"LowPassFilter sensorFilter({num[0]}, {num[1]}, {den[0]}, {den[1]});"

        parts = [
            '#include <Servo.h>',
            '#include <math.h>',
            '',
            self.generateDefines(),
            '',
            '',
            self.generateLowPassFilter(),
            '',
            self.generateController(),
            '',
            self.generateSetpointer(),
            '',
            '',
            self.generateHardwareClasses(),
            '',
            'KlaussServo myServo(11, MIN_OUT, MAX_OUT);',
            'UltrasonicSensor ultrasonic(8, 9);',
            '',
            filterInit,
            'Controller controller(INIT_X, INIT_U);',
            'Setpointer setpointer(INIT_X, TS_FILTER, DT);',
            '',
            'unsigned long startTime;',
            '',
            self.generateSetup(),
            '',
            self.generateLoop(),
        ]
        return '\n'.join(parts)

    def generateHardwareClasses(self):
        return """class KlaussServo {
private:
    Servo servo;

public:
    int pin;
    int currentAngle = 90;
    int targetAngle  = 90;
    int MIN_ANGLE, MAX_ANGLE;

    KlaussServo(int pin, int min, int max):
        pin(pin),
        MIN_ANGLE(min),
        MAX_ANGLE(max){}

    void setup(){
        servo.attach(pin);
        servo.write(currentAngle);
    }

    void setAngle(int angle){
        targetAngle = constrain(angle, MIN_ANGLE, MAX_ANGLE);
    }

    void update(){
        int passo = 5;
        if (abs(targetAngle - currentAngle) < passo)
            currentAngle = targetAngle;
        else
            currentAngle += (targetAngle > currentAngle) ? passo : -passo;
        servo.write(currentAngle);
    }

    void reset(){
        setAngle(90);
    }
};

class UltrasonicSensor{
private:
    int trigPin, echoPin;

public:
    float distance = 0;

    UltrasonicSensor(int trigPin, int echoPin):
        trigPin(trigPin),
        echoPin(echoPin) {}

    void setup(){
        pinMode(trigPin, OUTPUT);
        pinMode(echoPin, INPUT);
    }

    void update(){
        digitalWrite(trigPin, LOW);
        delayMicroseconds(2);
        digitalWrite(trigPin, HIGH);
        delayMicroseconds(10);
        digitalWrite(trigPin, LOW);

        long duration = pulseIn(echoPin, HIGH, 12000);
        distance = (duration > 0) ? (duration / 2.0) * 0.000343 : 4.0;
    }
};"""

    def generateSetup(self):
        return """void setup(){
    Serial.begin(9600);
    setpointer.update(0.15);

    sensorFilter.reset(INIT_X);
    myServo.setup();
    ultrasonic.setup();

    startTime = millis();
}"""

    def generateLoop(self):
        return """void loop() {
    const unsigned long dt = (unsigned long)(DT * 1000);
    static unsigned long lastStepTime = 0;
    unsigned long now = millis();

    if(now - lastStepTime < dt)
        return;

    lastStepTime = now;
    ultrasonic.update();
    float sensorValue = sensorFilter.update(ultrasonic.distance);

    float ref = setpointer.tick();
    float u   = controller.update(ref, sensorValue);

    int angle = (int) constrain(u, MIN_OUT, MAX_OUT);
    myServo.setAngle(angle);
    myServo.update();

    Serial.print("{\\\"time\\\":");
    Serial.print((now - startTime) / 1000.0, 3);
    Serial.print(",\\\"servo\\\":");
    Serial.print(myServo.currentAngle);
    Serial.print(",\\\"distance\\\":");
    Serial.print(sensorValue, 6);
    Serial.print(",\\\"raw\\\":");
    Serial.print(ultrasonic.distance, 6);
    Serial.print(",\\\"setpoint\\\":");
    Serial.print(setpointer.getTarget(), 6);
    Serial.print(",\\\"ref\\\":");
    Serial.print(setpointer.getRef(), 6);
    Serial.print(",\\\"u\\\":");
    Serial.print(u, 4);
    Serial.println("}");
}"""


if __name__ == '__main__':
    basePath = os.path.dirname(os.path.abspath(__file__))
    infoPath = os.path.join(basePath, '..', 'info.json')
    ctlDir = os.path.join(basePath, '..', 'Backup', 'Controller', 'model_1')
    ctlInfoPath = os.path.join(ctlDir, 'info.json')

    outputPath = os.path.join(basePath, '..', '..', 'Controller', 'Controller.ino')
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)

    generator = ArduinoCodeGenerator(infoPath, ctlInfoPath)
    code = generator.generateFull()

    with open(outputPath, 'w') as f:
        f.write(code)
        f.write('\n')

    print(f'Generated Controller.ino at {outputPath}')
    print(f'  filter num = {generator.info["sensor_filter"]["num"]}')
    print(f'  filter den = {generator.info["sensor_filter"]["den"]}')
    print(f'  PID num    = {generator.ctlInfo["num"]}')
    print(f'  PID den    = {generator.ctlInfo["den"]}')
