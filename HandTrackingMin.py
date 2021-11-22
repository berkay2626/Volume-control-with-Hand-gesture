import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

mp_cizim = mp.solutions.drawing_utils
mp_cizim_sekilleri = mp.solutions.drawing_styles
mp_el = mp.solutions.hands

cihazlar = AudioUtilities.GetSpeakers()
arayüz = cihazlar.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
ses = cast(arayüz, POINTER(IAudioEndpointVolume))

menzil = ses.GetVolumeRange()
minVol, maxVol, volBar, volPer = menzil[0], menzil[1], 400, 0

gKamera, yKamera = 640, 480 #g = genislik , y = yükseklik
kamera = cv2.VideoCapture(0)
kamera.set(3, gKamera)
kamera.set(4, yKamera)
 

 with mp_el.Hands(model_complexity = 0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
 
    while (True):

        basari, görüntü = kamera.read()

        görüntü = cv2.cvtColor(görüntü, cv2.COLOR_BGR2RGB)
        sonuc = hands.process(cv2.cvtColor(görüntü, cv2.COLOR_BGR2RGB))

        if sonuc.multi_hand_landmarks:
            for el_isareti in sonuc.multi_hand_landmarks:
                mp_cizim.draw_landmarks(
                    görüntü,
                    el_isareti,
                    mp_el.HAND_CONNECTIONS,
                    mp_cizim_sekilleri.get_default_hand_landmarks_style(),
                    mp_cizim_sekilleri.get_default_hand_connections_style()
                    )

        lmList = []
        if sonuc.multi_hand_landmarks:
            benimElim = sonuc.multi_hand_landmarks[0]
            for id, lm in enumerate(benimElim.landmark):
                h,w,c = görüntü.shape
                cx, cy = int(lm.x * w), int(lm.y *h)
                lmList.append([id, cx, cy])
                
            if len(lmList) != 0:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1],lmList[8][2]

                cv2.circle(görüntü, (x1,y1),15,(255,255,255))  
                cv2.circle(görüntü, (x2,y2),15,(255,255,255))  
                cv2.line(görüntü,(x1,y1),(x2,y2),(0,255,0),3)
            uzunluk = math.hypot(x2-x1,y2-y1)
            if uzunluk < 50:
                cv2.line(görüntü,(x1,y1),(x2,y2),(0,0,255),3)


            vol = np.interp(uzunluk, [50, 220], [minVol, maxVol])


            ses.SetMasterVolumeLevel(vol, None)
            volBar = np.interp(uzunluk, [50, 220], [400, 150])
            volPer = np.interp(uzunluk, [50, 220], [0, 100])


            cv2.rectangle(görüntü, (50, 150), (85, 400), (0, 0, 0), 3)
            cv2.rectangle(görüntü, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
            cv2.putText(görüntü, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 
                    1, (0, 0, 0), 3)


        cv2.imshow('El kontrol', görüntü)
        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
kamera.release()
