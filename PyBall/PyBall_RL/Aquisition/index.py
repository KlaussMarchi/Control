# pip install pandas
# pip install pyserial
# pip install matplotlib

import serial, ast, json
import serial.tools.list_ports
import pandas as pd
from time import sleep, time

def sendEvent(eventType, message, color='blue', delay=0.0):
    status = eventType
    
    pythonColors = {
        'blue' : '\033[34m',
        'red'  : '\033[31m',
        'green': '\033[32m',
        'white': '\033[0m', 
    }

    if eventType == 'success':
        status = True
        color  = 'blue'

    if eventType == 'error':
        status = False
        color  = 'red'

    if eventType == 'event':
        status = None
        color = 'green'

    status = True if eventType == 'success' else False if eventType == 'error' else None
    color  = pythonColors[color]
    reset  = pythonColors['white']
    
    print(f'{color}[{eventType}]{reset}', message)
    if delay > 0.0: sleep(delay)
    return status


class Device:
    device = None
    port   = None
    rate   = None
    data   = None

    def __init__(self, port=None, rate=9600, timeout=3.0):
        self.port = port
        self.rate = rate
        self.timeout = timeout
        self.last = None

    def connect(self, delay=2.5):
        self.port = self.port if self.port is not None else self.scan()
        sendEvent('event', f'trying connection: {self.port}')
        
        if self.device and self.device.is_open:
            return sendEvent('success', f'already connected')
        try:
            self.device = serial.Serial(self.port, self.rate, timeout=self.timeout)
            return sendEvent('success', 'device connected successfully', delay=delay)
        except Exception as error:
            self.device = None
            return sendEvent('error', error, delay=3.0)
    
    def reconnect(self):
        if not bool(self.device and self.device.is_open):
            self.connect()

    def disconnect(self):
        if not self.device:
            return  
        try:
            if self.device.is_open:
                self.device.close()
        except Exception as error:
            sendEvent('error', error)
        
    def getPorts(self):
        return [port for port in serial.tools.list_ports.comports()]
    
    def scan(self):
        ports  = self.getPorts()
        target = 0

        if not ports:
            return sendEvent('error', 'no port found')

        for i, port in enumerate(ports):
            if 'usb' in str(port).lower():
                target = i
        
        selected = str(ports[target]).strip()
        parts    = selected.split(' ')  

        if selected == '' or len(parts) == 0:
            return sendEvent('error', f'parts: {parts}')

        print('ports:', parts)
        return parts[0].strip()
    
    def send(self, msg, breakLine=True):
        msg = msg.strip()
        
        if breakLine:
            msg = msg + '\r\n'

        try:
            self.device.write(msg.encode())
            #sendEvent('python', f'sent: {msg.strip()}', 'red')
        except Exception as error:
            return sendEvent('error', error)
        
        return True
    
    def wait(self, timeout=5):
        startTime = time()

        while time() - startTime < timeout:
            if self.available():
                return True

        return False
    
    def get(self, timeout=7.0):
        startTime = time()
        response  = bytearray()

        try:
            while time() - startTime < timeout:
                size = self.device.in_waiting

                if size == 0:
                    continue
                
                newByte = self.device.read(size)

                if newByte in ['\r', '\n', '\t', '}']:
                    break

                response.extend(newByte)

            cleaned = response.decode('utf-8', errors='ignore').strip()
            print(cleaned)

            if '\n' in cleaned:
                cleaned = cleaned.split('\n')[0].strip()

            return cleaned
        except Exception as error:
            sendEvent('error', error)
            return None

    def getJson(self, timeout=5.0):
        try:
            old_timeout = self.device.timeout
            self.device.timeout = timeout
            line = self.device.readline()
            self.device.timeout = old_timeout

            if not line:
                return None
            
            cleaned = line.decode('utf-8', errors='ignore').strip()
            
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            
            if start != -1 and end != -1 and start < end:
                return json.loads(cleaned[start:end+1])
                
            return None
        except Exception as error:
            sendEvent('error', error)
            return None

    def available(self):
        if not self.device:
            return 0
        
        totalBytes = self.device.in_waiting
        return totalBytes if totalBytes > 0 else 0
    
    def getResponse(self, msg):
        if not self.send(msg):
            return None

        return self.get(10)

    def clear(self, delay=0):
        self.device.reset_input_buffer()
        sleep(0.5)

        while self.available():
            try:
                self.device.read()
            except:
                continue
        
        self.device.reset_input_buffer()
        sleep(0.5 + delay)

    def expect(self, target='OK', command=None, fail=None, timeout=5):
        startTime  = time()
        buffer     = str()
        timePassed = 0

        if command is not None:
            self.send(command)

        while timePassed < timeout:
            size = self.device.in_waiting
            timePassed = time() - startTime
            
            if command and timePassed > 0.95*timeout:
                self.send(command)
            
            if size == 0:
                continue
            
            buffer += self.device.read(size).decode('utf-8', errors='ignore')

            if target in buffer:
                return True
            
            if fail and fail in buffer:
                return False

        return None
    
    def request(self, value, timeout=5.0):
        self.send(value)
        startTime = time()

        while time() - startTime < timeout:
            if self.available():
                continue

            return self.get()
        
        return None
    
    def stream(self, command='$STREAM!'):
        self.send(command)
        sleep(1.0)

    def handle(self):
        if not self.available():
            return
        
        self.last = self.getJson()   


if __name__ == '__main__':
    device = Device()
    device.connect()
    device.clear()
    device.stream()

    startTime = time()
    lastTime  = -1
    data = []
    
    while time() - startTime < 500:
        device.handle()

        if not device.last:
            continue
        
        if device.last['time'] == lastTime:
            continue
        
        lastTime = device.last['time']
        print(device.last)
        data.append(device.last)


df = pd.DataFrame(data)
df.to_csv('DataBase.csv', index=None)
print(df)