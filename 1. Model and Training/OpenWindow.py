import tkinter as tk
from Window import Window

def main():
    root=tk.Tk()
    app=Window(master=root)
    app.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")