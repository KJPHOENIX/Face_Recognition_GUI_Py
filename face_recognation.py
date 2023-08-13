import cv2
import tkinter
from tkinter import *
from functools import partial
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from simple_facerec import SimpleFacerec


class App(tk.Tk):
    
    def __init__(self):
        super().__init__()
        
        self.title('Face_recognition')
        self.geometry('300x100')
        self.config(bg = "gray")

        self.label = ttk.Label(self, text='FACE RECOGNITION',font=("optima",12,"bold"),background="gray")
        self.label.pack()

        self.traine =Button(self,text="Traine",font=("optima",12),bg="green",activebackground="lightgreen")
        self.traine['command']=self.Train
        self.traine.place(x = 50,y = 50)

        self.rec =Button(self,text="Recognation",font=("optima",12),bg="red",activebackground="pink")
        self.rec['command']=self.reco
        self.rec.place(x = 180,y = 50)



    def Train(self):

        new= Toplevel(self)
        new.geometry("300x200")
        new.title("Training")
        new.config(bg = "gray")

        #new.self.title('Face_recognition')
        #self.geometry('500x300')

        self.name = Label(new,text = "Enter name :",font=("optima",12),bg="gray")
        self.name.place(x=10,y =50)

        self.nameentry =Entry(new,font=("optima",12))
        self.nameentry.place(x=110,y=50)

        self.button = Button(new, text='capture image', font=("optima",12),bg="green",activebackground="red")
        self.button['command'] = self.capture_image
        self.button.place(x=10,y = 80)

        

        self.mainloop()

    
    def capture_image(self):
        
        e = self.nameentry.get()
        camera = cv2.VideoCapture(0)
        for i in range(1):
            return_value, image = camera.read()
            cv2.imwrite(str(e)+'.png', image)
        del(camera)

    
    def reco(self):
        sfr = SimpleFacerec()
        sfr.load_encoding_images("F:/vscode/") #file location
        cap = cv2.VideoCapture(0)


        while True:
            ret, frame = cap.read()
            
            face_locations,face_names = sfr.detect_known_faces(frame)
            for face_loc, name in zip(face_locations,face_names):
                #print(face_loc)
                
                y1,x2,y2,x1 = face_loc[0],face_loc[1],face_loc[2],face_loc[3]
                
                cv2.putText(frame,name,(x1,y1 -10),cv2.FONT_HERSHEY_DUPLEX ,1,(0,0,0), 2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,200),4)
                

            cv2.imshow("frame",frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
  app = App()
  app.mainloop()
