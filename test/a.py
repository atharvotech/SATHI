import pyautogui


pyautogui.moveTo(100,100,duration=1.0)
pyautogui.sleep(1)
pyautogui.dragTo(1000,100, duration=1)
pyautogui.sleep(1)

pyautogui.moveTo(300,300,duration=1.0)
pyautogui.sleep(1)
pyautogui.press('win')
pyautogui.sleep(1)
pyautogui.typewrite('notepad', interval=0.1)
pyautogui.sleep(1)
pyautogui.press('enter')
pyautogui.sleep(1)
pyautogui.typewrite('Hello this is a test of pyautogui automation!',interval=0.1)