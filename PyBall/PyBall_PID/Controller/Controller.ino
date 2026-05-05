#include <Servo.h>
#include <math.h>

#define DT         0.045
#define OFFSET     3
#define TS_FILTER  15.0
#define SENSOR_MIN 400
#define SENSOR_MAX 900

#define MIN_OUT 0
#define MAX_OUT 255

#define INIT_X  650.0
#define INIT_U  127.5


class LowPassFilter {
  private:
    float num[2];
    float den[2];
    float Xn[2];
    float Yn[2];

  public:
    LowPassFilter(float n0, float n1, float d0, float d1) {
      num[0] = n0; num[1] = n1;
      den[0] = d0; den[1] = d1;
      for (int i = 0; i < 2; i++) {
        Xn[i] = 0.0;
        Yn[i] = 0.0;
      }
    }

    void reset(float value) {
      for (int i = 0; i < 2; i++) {
        Xn[i] = value;
        Yn[i] = value;
      }
    }

    float update(float inputValue) {
      for (int n = 1; n > 0; n--) {
        Xn[n] = Xn[n - 1];
        Yn[n] = Yn[n - 1];
      }
      Xn[0] = inputValue;

      float out = 0.0;
      for (int i = 0; i < 2; i++)
        out += Xn[i] * num[i];
      for (int i = 1; i < 2; i++)
        out -= Yn[i] * den[i];

      Yn[0] = out;
      return out;
    }
};

class Controller {
  private:
    float num[3];
    float den[3];
    float Xn[3];
    float Yn[3];

  public:
    Controller(float initX, float initU) {
      num[0] = 241.14494304805794; num[1] = -156.55695211422346; num[2] = -76.38438902925166;
      den[0] = 1.0; den[1] = -0.2109192880696864; den[2] = -0.7890807119303136;
      
      for (int i = 0; i < 3; i++) {
        Xn[i] = 0.0;
        Yn[i] = initU;
      }
    }

    float update(float setpoint, float sensorValue) {
        float error = setpoint - sensorValue;
        
        for (int n = 2; n > 0; n--) {
            Xn[n] = Xn[n - 1];
            Yn[n] = Yn[n - 1];
        }
        Xn[0] = error;

        float out = 0.0;
        for (int i = 0; i < 3; i++)
            out += Xn[i] * num[i];
            
        for (int i = 1; i < 3; i++)
            out -= Yn[i] * den[i];

        if (out < MIN_OUT) out = MIN_OUT;
        if (out > MAX_OUT) out = MAX_OUT;

        Yn[0] = out;
        return out;
    }
};

class Setpointer {
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
};


class KlaussServo {
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
};

KlaussServo myServo(11, MIN_OUT, MAX_OUT);
UltrasonicSensor ultrasonic(8, 9);

LowPassFilter sensorFilter(0.4589418031905226, 0.4589418031905226, 1.0, -0.08211639361895486);
Controller controller(INIT_X, INIT_U);
Setpointer setpointer(INIT_X, TS_FILTER, DT);

unsigned long startTime;

void setup(){
    Serial.begin(9600);
    setpointer.update(0.15);

    sensorFilter.reset(INIT_X);
    myServo.setup();
    ultrasonic.setup();

    startTime = millis();
}

void loop() {
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

    Serial.print("{\"time\":");
    Serial.print((now - startTime) / 1000.0, 3);
    Serial.print(",\"servo\":");
    Serial.print(myServo.currentAngle);
    Serial.print(",\"distance\":");
    Serial.print(sensorValue, 6);
    Serial.print(",\"raw\":");
    Serial.print(ultrasonic.distance, 6);
    Serial.print(",\"setpoint\":");
    Serial.print(setpointer.getTarget(), 6);
    Serial.print(",\"ref\":");
    Serial.print(setpointer.getRef(), 6);
    Serial.print(",\"u\":");
    Serial.print(u, 4);
    Serial.println("}");
}
