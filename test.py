import wx
import wx.lib.buttons as buttons

class Main(wx.Frame):
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, id=-1, title=title, size=(300, 300))

        self.panel = wx.Panel(self, wx.ID_ANY, wx.DefaultPosition, size=(300, 300))

        firstImage = ButtonImage(self, self.panel, 'img/0.png', (20,20))
        secondImage = ButtonImage(self, self.panel, 'img/1.png', (20, 135))

        self.Layout()

class ButtonImage():
    def __init__(self, parent, panel, nameImage, pos):
        self.panel = panel

        self.bmp = wx.Bitmap(nameImage, wx.BITMAP_TYPE_ANY)

        ### search for a piece of the maximum position (limit boundary)
        self.maxPiecePositionX = self.panel.GetSize()[0] - self.bmp.GetSize()[0]
        self.maxPiecePositionY = self.panel.GetSize()[1] - self.bmp.GetSize()[1]

        self.bmapBtn = wx.BitmapButton(self.panel, id=wx.ID_ANY, bitmap=self.bmp, style=wx.NO_BORDER, pos=pos)

        #self.bmapBtn.Bind(wx.EVT_ENTER_WINDOW, self.EnterButton, self.bmapBtn)
        #self.bmapBtn.Bind(wx.EVT_LEAVE_WINDOW, self.LeaveButton, self.bmapBtn)

        self.bmapBtn.Bind(wx.EVT_LEFT_DOWN, self.OnClickDown, self.bmapBtn)
        self.bmapBtn.Bind(wx.EVT_LEFT_UP, self.OnClickUp, self.bmapBtn)
        self.bmapBtn.Bind(wx.EVT_MOTION, self.MoveButton, self.bmapBtn)

        #self.bmapBtn.Bind(wx.EVT_MOVE, self.MoveButtonTest, self.bmapBtn)

        self.hold = 0
        self.holdPosition = (0, 0)

    def EnterButton(self, event):
        pass

    def LeaveButton(self, event):
        self.hold = 0

    def OnClickDown(self, event):
        obj = event.GetEventObject()
        self.hold = 1
        self.holdPosition = (event.GetX(), event.GetY())

    def OnClickUp(self, event):
        self.hold = 0

    def MoveButton(self, event):
        deltaX, deltaY = 0, 0

        if self.hold:
            deltaX = event.GetPosition()[0] - self.holdPosition[0]
            deltaY = event.GetPosition()[1] - self.holdPosition[1]

            newPositionX = self.bmapBtn.GetPosition()[0] + deltaX
            newPositionY = self.bmapBtn.GetPosition()[1] + deltaY

            if (0 < newPositionX < self.maxPiecePositionX) and (0 < newPositionY < self.maxPiecePositionY):
                self.bmapBtn.SetPosition((newPositionX, newPositionY))
            else:
                self.holdPosition = self.holdPosition[0] + deltaX, self.holdPosition[1] + deltaY
            self.bmapBtn.Refresh()

app = wx.App()
frame = Main(None, u"Game")
frame.Show()
app.MainLoop()