### stm32 IDE를 이용한 방법
```c
HAL_GPIO_WritePin(GPIOB, GPIO_PIN_12|GPIO_PIN_15|GPIO_PIN_5, GPIO_PIN_RESET);
HAL_GPIO_WritePin(GPIOC, GPIO_PIN_6|GPIO_PIN_7, GPIO_PIN_RESET);
void StartTask02(void const * argument){ 
  for(;;){
      app();  
      osDelay(1);
      }  
}
void StartTask03(void const * argument){
   for(;;){  
       control();
       osDelay(10);
  }
}
```


![image](https://github.com/dnfm257/cctv_ctrl/assets/143377935/0f4c9c86-6a6d-4bfc-8326-44c9f69d8718)
