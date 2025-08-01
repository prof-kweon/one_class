#include<Servo.h> //Servo 라이브러리를 추가
#include <Servo.h>

Servo servo; 			//servo로 서보모터 함수 활용함

void setup() {
  servo.attach(2);		//서보모터핀은 디지털 2번핀
}

void loop() {
  int degree[8] = {0, 45, 90, 135, 180, 135, 90, 45}; //각도 조절을 위한 배열
  for(int a = 0; a < 8; a++)
  {
    servo.write(degree[a]); 	//배열을 이용해서 0~180도를 45도씩 돌아서 다시 0도로 돌아옴
    delay(1000);				//1초간 대기
  }
}