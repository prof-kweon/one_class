void setup() {
  // put your setup code here, to run once:
  pinMode(8, OUTPUT);    // LED 출력
  pinMode(7, INPUT);     // 스위치 버튼 입력
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  int value = digitalRead(7);   // 입력핀의 값을 읽어서 변수에 저장
  Serial.println(value);

  if(value == HIGH) {
    digitalWrite(8, HIGH);  // LED ON
  }
  else { 
    // delay(1000);             // 1초 기다림
    digitalWrite(8, LOW);   // LED OFF
  }
}
