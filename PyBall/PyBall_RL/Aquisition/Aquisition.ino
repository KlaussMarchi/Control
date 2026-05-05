#include <Servo.h>
#define MIN_OUT 45
#define MAX_OUT 135

class KlaussServo {
private:
    Servo servo;
    unsigned long lastMoveTime = 0;

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
        int passo = 5; // velocity

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

class SignalGenerator{
private:
    int min, max;
    unsigned long lastChange = 0;
    bool firstRun = true;

public:
    const unsigned long period = 3000;
    float output;

    SignalGenerator(int min, int max) {
        this->min = min;
        this->max = max;
        this->output = min;
    }

    void update() {
        unsigned long now = millis();
        
        if(now - lastChange >= period || firstRun) {
            lastChange = now;
            firstRun = false;
            output = random(min, max+1);
            period = random(500, 5000);
        }
    }
};

SignalGenerator generator(MIN_OUT, MAX_OUT);
KlaussServo myServo(11, MIN_OUT, MAX_OUT);
UltrasonicSensor ultrasonic(8, 9);
unsigned long startTime;

void setup() {
    Serial.begin(9600);
    
    myServo.setup();
    ultrasonic.setup();

    while(!Serial.available())
        continue;
    
    randomSeed(analogRead(0));
    startTime = millis();
}

void loop() {
    ultrasonic.update();
    generator.update();
    
    myServo.setAngle(generator.output);
    myServo.update();

    Serial.print("{\"time\":");
    Serial.print((millis() - startTime) / 1000.0, 3);
    Serial.print(",\"servo\":");
    Serial.print(myServo.currentAngle);
    Serial.print(",\"distance\":");
    Serial.print(ultrasonic.distance, 3);
    Serial.println("}");
}