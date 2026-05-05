#include <Servo.h>
#include <math.h>

#define DT         0.048
#define OFFSET     22
#define TS_FILTER  15.0
#define SENSOR_MIN 0
#define SENSOR_MAX 0.3

#define MIN_OUT 45
#define MAX_OUT 135

#define INIT_X  0.00
#define INIT_U  90.0

#define N_STATES   5
#define N_FEATURES 14


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

class StatesBuffer {
  private:
    float buffer[N_STATES];
    int size;

  public:
    StatesBuffer(int sz, float initial = 0.0) : size(sz) {
        for (int i = 0; i < size; i++)
            buffer[i] = initial;
    }

    void update(float value) {
        for (int i = size - 1; i > 0; i--)
            buffer[i] = buffer[i - 1];
        buffer[0] = value;
    }

    float get(int idx) {
        return buffer[idx];
    }
};

class Model {
  private:
    const float mean[N_FEATURES] = {
        94.33684108901332,
        94.33056574628307,
        94.32477312222437,
        94.31946321683722,
        0.11645650665051889,
        0.11647089920611557,
        0.11648444953476472,
        0.1164966370246822,
        0.11650686246835111,
        -0.00021342819649479216,
        -0.00022444512821668186,
        -0.00023422672858617918,
        -0.00024236121925877183,
        -0.0002486758437582931
    };

    const float scale[N_FEATURES] = {
        24.158579687712535,
        24.157757208298058,
        24.15639704049569,
        24.154599363692515,
        0.07868482688923085,
        0.07867880427828887,
        0.07867237999443955,
        0.07866550590168125,
        0.07865841558249823,
        0.05388355171706332,
        0.05389517140642046,
        0.05390432324753924,
        0.05391064436823212,
        0.05391444607314255
    };

    const float coef[N_FEATURES] = {
        22.59665120731033,
        -24.67143669761124,
        1.5568348657607096,
        0.43763141196633626,
        0.8085142456098389,
        -0.8557449497919198,
        0.02164542946190995,
        0.5620503551189784,
        -0.5296082286279913,
        0.3858188391195192,
        -0.298084157746644,
        -0.2632753365747589,
        0.5833870539475234,
        -0.4159437279800402
    };

    const float intercept = 0.00675806140180611;

    float standardize(float value, int idx) {
        return (value - mean[idx]) / scale[idx];
    }

  public:
    float predict(float features[N_FEATURES]) {
        float result = intercept;
        for (int i = 0; i < N_FEATURES; i++)
            result += coef[i] * standardize(features[i], i);
        return result;
    }
};

class Controller {
  private:
    Model model;
    StatesBuffer actuator;
    StatesBuffer sensor;
    StatesBuffer errorBuf;

  public:
    Controller(float initX, float initU)
        : actuator(N_STATES, initU),
          sensor(N_STATES, initX),
          errorBuf(N_STATES, 0.0) {}

    float update(float setpoint, float sensorValue) {
        sensor.update(sensorValue);
        errorBuf.update(setpoint - sensor.get(0));

        float features[N_FEATURES];
        int idx = 0;

        for (int i = 1; i < N_STATES; i++)
            features[idx++] = actuator.get(i - 1);

        for (int i = 0; i < N_STATES; i++)
            features[idx++] = sensor.get(i);

        for (int i = 0; i < N_STATES; i++)
            features[idx++] = errorBuf.get(i);

        float deltaU = model.predict(features);
        float response = actuator.get(0) + deltaU;

        if (response < MIN_OUT) response = MIN_OUT;
        if (response > MAX_OUT) response = MAX_OUT;

        actuator.update(response);
        return response;
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

LowPassFilter sensorFilter(0.18446880939648136, 0.1844688093964813, 1.0, -0.6310623812070374);
Controller controller(INIT_X, INIT_U);
Setpointer setpointer(INIT_X, TS_FILTER, DT);

unsigned long startTime;

void setup(){
    Serial.begin(9600);
    setpointer.update(0.10);

    sensorFilter.reset(INIT_X);
    myServo.setup();
    ultrasonic.setup();

    startTime = millis();
}

void loop() {
    const unsigned long dt = 0; // (unsigned long)(DT * 1000);
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
    Serial.print(sensorValue, 3);
    Serial.println("}");
}
