# STM32 NUCLEO-F429ZI를 assembly로 제어해본 내용
## 사용이유
### 1
linux환경에서는 upgrade나 ubuntu version에 따라 일시적으로 stm32 ide프로그램이 작동하지 않을 수 있음
그러한 상황이 왔을때를 대비한 ide를 사용하지 않고 코딩하는 방법이 필요
다양한 방법 중 최소한의 파일이 필요한 assembly어를 선택하고 이를 시도해봄
### 2
경찰청 교통신호제어기 표준규격서 R27(2021.10.05)에 의한 규격을 보면은 신호등의 제어와 관련된 adress 주소등에 대한 상세한 설명이 있음
이를 참고하여 제작해보고자 진행함


## datasheet 

#### 경찰청 교통신호제어기 표준규격서 R27(2021.10.05)
https://www.koroad.or.kr/main/board/21/22366/board_view.do?&cp=3&listType=list&bdOpenYn=Y&bdNoticeYn=N

#### 연산기호에 대한 내용
(QRC0006_UAL16.pdf)

https://github.com/sysplay/bbb/blob/master/Docs/QRC0006_UAL16.pdf  

#### stm32 f429zi에 대한 datasheet
(dm00031020-stm32f405-415-stm32f407-417-stm32f427-437-and-stm32f429-439-advanced-arm-based-32-bit-mcus-stmicroelectronics.pdf)

https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwit0IHP6-CCAxUsslYBHflPAgsQFnoECA0QAQ&url=https%3A%2F%2Fwww.st.com%2Fresource%2Fen%2Freference_manual%2Fdm00031020-stm32f405-415-stm32f407-417-stm32f427-437-and-stm32f429-439-advanced-arm-based-32-bit-mcus-stmicroelectronics.pdf&usg=AOvVaw2x8tbTRz8d9PfqXBk3qZ74&opi=89978449  

#### tim핀 설정을 위한 cortex-m4 datasheet
(pm0214-stm32-cortexm4-mcus-and-mpus-programming-manual-stmicroelectronicsㄴ.pdf)

https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjq0piZ7OCCAxWsr1YBHYBYDbkQFnoECBAQAQ&url=https%3A%2F%2Fwww.st.com%2Fresource%2Fen%2Fprogramming_manual%2Fpm0214-stm32-cortexm4-mcus-and-mpus-programming-manual-stmicroelectronics.pdf&usg=AOvVaw0Wm2Kvb6UMOUO7rWx3Zup9&opi=89978449


#### MB1137.pdf  => f429zi의 pin에 대한 설정 파일  

## assembly 핀에 대한 설명(start_1124.asm)
register의 adress의 값에 대한 설명  

## 내장 LED제어(111d.asm)
led의 pb7에 연결되어 있는 LD2를 제어한 assembly코드

## 3개의 gpio pin을 제어하여 3개의 외부 led 점등(112202.asm)
pb라인의 9,12,13,15를 이용한 delay led점등

## tim을 이용한 switching 점등(112401.asm)
직점 triger를 설정해서 flag 횟수에 따라 점등이 변경되는 코드(아직 수정중)

## 환경설정 및 정보에 대한 사진들(qqqww.elf는 stm32로 만든 동일한 동작을 하는 프로그램)
![스크린샷 2023-11-25 13-25-32](https://github.com/dnfm257/cctv_ctrl/assets/143377935/1b274bd7-9089-4066-b5cf-fcbb308c9064)
![스크린샷 2023-11-25 13-19-20](https://github.com/dnfm257/cctv_ctrl/assets/143377935/4d05dcfd-86b7-445d-a6c0-cd8176951d9e)
![스크린샷 2023-11-25 13-14-11](https://github.com/dnfm257/cctv_ctrl/assets/143377935/e2d24b90-9973-406b-9fa2-1e8dee553937)
![스크린샷 2023-11-25 13-50-29](https://github.com/dnfm257/cctv_ctrl/assets/143377935/12b9460f-6ba4-45ca-b2b2-ef0fb7cc456a)
![스크린샷 2023-11-27 14-18-17](https://github.com/dnfm257/cctv_ctrl/assets/143377935/090bf48d-04f3-40ce-9076-e364a6f46a72)

