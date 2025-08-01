int echo = 2, trig = 3; //echo핀은 2번, trig핀은 3번으로 정함

void setup()
{
  Serial.begin(9600); 		//시리얼 모니터를 사용하기 위해 선언
  pinMode(trig, OUTPUT);	//trig핀은 초음파를 발사함
  pinMode(echo, INPUT);		//echo핀은 반사된 초음파를 들음
}

void loop()
{
  float duration, distance; 				  //초음파가 되돌아온 시간과 길이를 소수형 변수로 저장
  
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW); 					  //초음파를 10us간 발사함
  
  duration = pulseIn(echo, HIGH); 			  //발사된 초음파의 길이를 duration에 저장
  
  distance = ((340 * duration) / 10000) / 2;  //cm로 계산, 두자리수 소수까지 출력

  Serial.print(distance);
  Serial.println("cm"); 	//계산된 값을 시리얼 모니터로 보여줌
  delay(500); 			
}
