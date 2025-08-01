void setup() {
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    char val = Serial.read();
    Serial.println(val);  // Console.println은 존재하지 않음, Serial.println이 맞습니다
    if (val == 'R') {
      digitalWrite(13, HIGH); // Rock → 켜기
    } else {
      digitalWrite(13, LOW);  // Others → 끄기
    }
  }
}
