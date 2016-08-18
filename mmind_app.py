"""
    mmind_app.py

    This module is the main GUI-based app to play mastermind
    and watch the computer play various strategies.
"""

import wx
import wx.grid as gridlib
import numpy as np
import mastermind
import os

""" The following code snippet corrects filepaths """
app = False
filename, filetype = ([x.strip() 
    for x in os.path.basename(__file__).split('.')])
if app==True:
    path = filename+".app/Contents/MacOS/"
else:
    path = ""

########################################################################
class StartPanel(wx.Panel):
    """
        Panel class for the start panel that is displayed first
        upon opening the game
    """
    #----------------------------------------------------------------------
    def __init__(self, parent):
        """ Constructor creates the panel and displays some text """
        wx.Panel.__init__(self, parent=parent)
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        label = "Filename: %s Type: %s\n\n" % (filename, filetype)
        label += "Working Dir:\n%s\n\n" % str(os.getcwd())
        label += "Data path:\n%s\n\n" % path
        label += "Welcome to Mastermind! Click 'New' to Start New Game"
        newGameTxt = wx.StaticText(self, label=label)
        mainSizer.Add(newGameTxt, flag=wx.LEFT|wx.TOP, border=10)
        mainSizer.Add((-1, 10))
        self.SetSizer(mainSizer)


########################################################################
class NewGame(wx.Panel):
    """ New game panel to set up a new mastermind game """

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)

        png = wx.Image(path+'img/0.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        font = wx.Font(pointSize=16, weight=wx.BOLD, style=wx.NORMAL,
            family=wx.SYS_SYSTEM_FONT)

        mainSizer = wx.BoxSizer(wx.VERTICAL)
        newGameTxt = wx.StaticText(self, label="Please set up your game:")
        newGameTxt.SetFont(font)
        mainSizer.Add(newGameTxt, flag=wx.LEFT|wx.TOP, border=10)
        mainSizer.Add((-1, 10))

        #------- INPUT FIELDS
        iptbox = wx.BoxSizer(wx.HORIZONTAL)
        iptlabel = wx.StaticText(self, label="Code length:")
        self.clengthfield = wx.TextCtrl(self, value="3", size=(50, 25))

        iptbox.Add(iptlabel, flag=wx.LEFT|wx.TOP, border=0)
        iptbox.Add(self.clengthfield, flag=wx.LEFT|wx.RIGHT, border=10)
        mainSizer.Add(iptbox, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=10)

        #------- PEGBOX FOR CODE JAR
        pegbox = wx.BoxSizer(wx.HORIZONTAL)
        nPegs = 8
        fgs = wx.FlexGridSizer(2, nPegs, 5, 5)
        self.pegtc = [wx.TextCtrl(self, size=(30, 25), 
            style=wx.TE_PROCESS_ENTER, value='0') 
            for i in np.arange(nPegs)]
        pegtl = [wx.StaticBitmap(self, -1, 
            wx.Image(path+'img/%d.png' % i, wx.BITMAP_TYPE_ANY).ConvertToBitmap(), 
            size=(png.GetWidth(), png.GetHeight())) 
            for i in np.arange(nPegs)]
        fgs.AddMany(np.array([pegtl, self.pegtc]).flatten())
        [self.pegtc[i].SetValue('1') for i in np.arange(4)]

        peglabel = wx.StaticText(self, label="Code Jar:")
        mainSizer.Add(peglabel, flag=wx.LEFT|wx.TOP, border=10)
        pegbox.Add(fgs, proportion=1, 
            flag=wx.LEFT|wx.RIGHT|wx.EXPAND, border=0)
        mainSizer.Add(pegbox, flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP, 
            border=10)

        #------- SELECT PLAYING AGENT AND STRATEGY
        mainSizer.Add((-1, 20))
        agenttxt = wx.StaticText(self, label="Select Agent:")
        mainSizer.Add(agenttxt, flag=wx.LEFT, border=10)
        self.rbtns = []
        self.rbtns.append(wx.RadioButton(self, -1, 'no agent (human player)', #0
            style=wx.RB_GROUP))
        self.rbtns.append(wx.RadioButton(self, -1, 'random feasible'))  #1
        self.rbtns.append(wx.RadioButton(self, -1, 'pure probability')) #2
        self.rbtns.append(wx.RadioButton(self, -1, 'pure information gain')) #3
        self.rbtns.append(wx.RadioButton(self, -1, 'mixed strategy')) #4
        
        for btn in self.rbtns: mainSizer.Add(btn, flag=wx.LEFT, border=10)
        self.rbtns[0].SetValue(True)

        #info_gain_parametrization
        mainSizer.Add((-1, 6))
        SM_param = wx.BoxSizer(wx.HORIZONTAL)
        orderTxt = wx.StaticText(self, label="--> order (r):")
        SM_param.Add(orderTxt)
        self.degreeIpt = wx.TextCtrl(self, value="1.0", size=(40, 25))
        SM_param.Add(self.degreeIpt)
        degreeTxt = wx.StaticText(self, label="degree (t):")
        SM_param.Add(degreeTxt, flag=wx.LEFT, border=10)
        self.orderIpt = wx.TextCtrl(self, value="1.0", size=(40, 25))
        SM_param.Add(self.orderIpt)
        mainSizer.Add(SM_param, flag=wx.LEFT, border=27)

        #trade-off parameter
        mainSizer.Add((-1, 6))
        mix_param = wx.BoxSizer(wx.HORIZONTAL)
        mixTxt = wx.StaticText(self, label="--> mix parameter:")
        self.mixIpt = wx.TextCtrl(self, value="0.5", size=(40, 25))
        mix_param.Add(mixTxt)
        mix_param.Add(self.mixIpt)
        mainSizer.Add(mix_param, flag=wx.LEFT, border=27)
        mixTxt2 = wx.StaticText(self, label="(0: 100% inf., 1: 100% prob.)")
        mainSizer.Add(mixTxt2,flag=wx.LEFT, border=50)

        #------- start game button
        mainSizer.Add((-1, 10))
        gamebtn = wx.Button(self, -1, 'Start Game')
        mainSizer.Add(gamebtn, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=10)
        self.Bind(wx.EVT_BUTTON, parent.createNewGame, gamebtn)

        #------- layout
        self.SetSizer(mainSizer)
    

########################################################################
class PlayGame(wx.Panel):
    """ Main game panel where the game is played via drag-and-drop """ 

    #----------------------------------------------------------------------
    def __init__(self, parent, **kwargs):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)
        self.parent = parent

        #-------- initialize game
        self.colorIDX = np.nonzero(kwargs['codejar'])[0]
        kwargs['codejar'] = (np.array(kwargs['codejar'])
            [np.nonzero(kwargs['codejar'])])
        self.game = mastermind.Game(**kwargs)
        self.game.initialize()

        #-------- determine mode (human or AI?)
        if not kwargs['mode']==0:
            print "Computer player"
            self.human_player = False
            self.agent = mastermind.AppAgent(
                game=self.game, mode=kwargs['mode'],
                r=kwargs['r'], t=kwargs['t'])
        else: 
            print "Human player"
            self.human_player = True

        mainSizer = wx.BoxSizer(wx.VERTICAL)
        newGameTxt = wx.StaticText(self, label="Drag and Drop marbles" \
            " to desired position")
        mainSizer.Add(newGameTxt, flag=wx.LEFT|wx.TOP, border=10)
        mainSizer.Add((-1, 10))

        #-------- load images
        self.black_token = wx.Image(path+'img/b.png', 
            wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.white_token =wx.Image(path+'img/w.png', 
            wx.BITMAP_TYPE_ANY).ConvertToBitmap()

        #-------- initialize stats window
        self.parent.statsFrame.InitializeStats(panel=self)
        self.UpdateStatistics()

        #-------- main game panel
        contentSizer = wx.BoxSizer(wx.HORIZONTAL)
        bgrd = wx.Image(path+'img/h.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        fgs = wx.FlexGridSizer(self.game.maxguess, 
            int(self.game.codelength), 6, 6)

        self.pegsbgrd = [wx.StaticBitmap(self, -1, 
            bgrd, size=(bgrd.GetWidth(), bgrd.GetHeight())) 
            for i in np.arange(self.game.maxguess*self.game.codelength)]

        fgs.AddMany(self.pegsbgrd)
        contentSizer.Add(fgs, proportion=1, 
            flag=wx.LEFT, border=0)
    
        ln = wx.StaticLine(self, -1, style=wx.LI_VERTICAL, size=(5,230))
        contentSizer.Add((6, -1))
        contentSizer.Add(ln, flag=wx.TOP)
        contentSizer.Add((6, -1))

        #-------- feedback panel
        fbgrid = wx.FlexGridSizer(self.game.maxguess, 
            self.game.codelength, 16, 6)
        fbo = wx.Image(path+'img/o.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.fbs = [wx.StaticBitmap(self, -1, 
            fbo, size=(fbo.GetWidth(), fbo.GetHeight())) 
            for i in np.arange(self.game.maxguess*self.game.codelength)]
        fbgrid.AddMany(self.fbs)
        
        contentSizer.Add(fbgrid, proportion=1, 
            flag=wx.LEFT | wx.TOP, border=4)

        mainSizer.Add(contentSizer, proportion=1, 
            flag=wx.LEFT, border=10)
        mainSizer.Add((-1, 20))

        #--------- guess btn
        guessbtn = wx.Button(self, -1, 'Make Guess')
        mainSizer.Add(guessbtn, flag=wx.LEFT|wx.RIGHT|wx.TOP, border=10)
        self.Bind(wx.EVT_BUTTON, self.MakeGuess, guessbtn)

        #-------- pegbox
        png = wx.Image(path+'img/0.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        pegbox = wx.FlexGridSizer(1, self.game.Ncolors, 5, 5)
        
        self.pegtl = [self.Marble(self, id=i, bitmap=wx.Image(path+'img/%d.png' % 
            self.colorIDX[i], wx.BITMAP_TYPE_ANY).ConvertToBitmap()) 
                for i in np.arange(self.game.Ncolors)]  
        pegbox.AddMany(self.pegtl)

        peglabel = wx.StaticText(self, label="Peg Box:")
        mainSizer.Add(peglabel, flag=wx.LEFT|wx.TOP, border=10)
        mainSizer.Add((-1, 10))
        mainSizer.Add(pegbox, proportion=1, 
            flag=wx.LEFT|wx.RIGHT|wx.EXPAND, border=10)
        mainSizer.Add((-1, 10))

        #-------- layout
        self.SetSizer(mainSizer)
        self.current_guess = [-1] * self.game.codelength
    
    #----------------------------------------------------------------------
    def UpdateStatistics(self):
        #first check if stats window is already open!
        if not self.parent.statsFrame.IsShown():
            return
        if self.parent.statsFrame.step != self.game.step:
            print "Update stats ..."
            pos, col, misc = self.game.compute_console_statistics()
            self.parent.statsFrame.UpdateStats(pos, col, misc)
            self.parent.statsFrame.step = self.game.step

    def MakeGuess(self, event):
        guessbtn = event.GetEventObject()
        # first determine computer guess 
        if not self.human_player:
            guessbtn.Disable() #disable button while computing
            self.current_guess = self.agent.compute_guess() #get computer guess
            print self.current_guess
            for i, idx in enumerate(self.get_active_zones()[0]):
                img = wx.Image(path+'img/%d.png' % 
                    self.colorIDX[self.current_guess[i]-1], 
                    wx.BITMAP_TYPE_ANY).ConvertToBitmap()
                self.pegsbgrd[idx].SetBitmap(img)
            self.Refresh()

        # make guess and get feedback
        if -1 in self.current_guess:    #check if guess complete
            wx.MessageBox('Guess not valid!', 'Error', 
                wx.OK | wx.ICON_EXCLAMATION) #wx.ICON_ERROR
        else:
            feedback = self.game.guess(combination=self.current_guess)
            # update stats
            self.UpdateStatistics()
            self.current_guess = [-1] * self.game.codelength
            startIdx = (self.game.codelength * (self.game.maxguess - 
                self.game.step - 1) + self.game.codelength)
            for pos in np.arange(feedback['position']):
                self.fbs[startIdx].SetBitmap(self.black_token)
                startIdx += 1
            for col in np.arange(feedback['color']):
                self.fbs[startIdx].SetBitmap(self.white_token)
                startIdx += 1
        if not guessbtn.IsEnabled(): #enable button if disabled
            guessbtn.Enable() 

        if self.game.end: #disable button when game is over
            guessbtn.Disable()

    def get_active_zones(self):
        positions = []
        idxs = []
        for i in np.arange(self.game.codelength):
            idx = (self.game.codelength * 
                (self.game.maxguess - self.game.step - 1) + i)
            idxs.append(idx)
            positions.append(self.pegsbgrd[idx].GetPosition().Get())
        return idxs, positions

    #----------------------------------------------------------------------
    class Marble(wx.StaticBitmap):
        def __init__(self, *args, **kwargs):
            png = wx.Image(path+'img/0.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            kwargs['size'] = (png.GetWidth(), png.GetHeight())
            wx.StaticBitmap.__init__(self, *args, **kwargs)
            self.codeid = kwargs['id']
            self.parent = args[0]
            self.hold = 0
            self.firstClick = True
            
            if self.parent.human_player:
                self.Bind(wx.EVT_LEFT_DOWN, self.OnClickDown, self)
                self.Bind(wx.EVT_LEFT_UP, self.OnClickUp, self)
                self.Bind(wx.EVT_MOTION, self.MoveMarble, self)

        def same_position(self, point1, point2, d=15):
            if (abs(point1[0] - point2[0]) < d and 
                abs(point1[1] - point2[1]) < d):
                return True
            else: return False

        def OnClickDown(self, event):
            # get position when first clicked
            if self.firstClick:
                self.originalPos = self.GetPosition()
                self.firstClick = False
            if not self.parent.game.end: #disable when game has ended
                self.hold = 1
            self.holdPosition = (event.GetX(), event.GetY())

        def OnClickUp(self, event):
            pos = self.GetPosition()
            self.hold = 0
            idxs, zones = self.parent.get_active_zones()
            for i, target in enumerate(zones):
                if self.same_position(pos, target):
                    img = wx.Image(path+'img/%d.png' % 
                        self.parent.colorIDX[self.codeid], 
                        wx.BITMAP_TYPE_ANY).ConvertToBitmap()
                    self.parent.pegsbgrd[idxs[i]].SetBitmap(img)
                    self.parent.current_guess[i] = self.codeid+1
            self.SetPosition(self.originalPos)
            self.Refresh()

        def MoveMarble(self, event):
            deltaX, deltaY = 0, 0
            if self.hold:
                deltaX = event.GetPosition()[0] - self.holdPosition[0]
                deltaY = event.GetPosition()[1] - self.holdPosition[1]
                newPositionX = self.GetPosition()[0] + deltaX
                newPositionY = self.GetPosition()[1] + deltaY

                # snippet can be used to prevent marbles from 
                # being dragged out of the panel window
                # if (0 < newPositionX < self.maxPiecePositionX) and 
                #   (0 < newPositionY < self.maxPiecePositionY):
                #     self.SetPosition((newPositionX, newPositionY))
                # else:
                #     self.holdPosition = self.holdPosition[0] + deltaX, 
                # self.holdPosition[1] + deltaY
                
                self.SetPosition((newPositionX, newPositionY))
                self.Refresh()


########################################################################
class StatsApp(wx.Frame):
    """
        Statistics Widget: Used to initialize computation and 
        visualization of game statistics
    """

    class PositionStats(wx.Panel):
        """ This panel is for the position statistics """ 
        def __init__(self, parent, **kwargs):
            """Constructor"""
            wx.Panel.__init__(self, parent=parent)
            self.parent = parent
            Ncolors = self.parent.gamePanel.game.Ncolors
            codelength = self.parent.gamePanel.game.codelength
            
            posSizer = wx.BoxSizer(wx.VERTICAL)
            font = wx.Font(pointSize=16, weight=wx.BOLD, style=wx.NORMAL,
                family=wx.SYS_SYSTEM_FONT)
            posText = wx.StaticText(self, label="Position")
            posText.SetFont(font)
            posSizer.Add(posText, flag=wx.LEFT, border=50)

            #-------- pegbox
            png = wx.Image(path+'img/h.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            posGrid = wx.FlexGridSizer(Ncolors+1, codelength+1, 5, 5)            
            self.posCont = np.empty(shape=(Ncolors+1, codelength+1), 
                dtype=object)
            
            for (i,j),_ in np.ndenumerate(self.posCont):
                self.posCont[i][j] = (wx.StaticText(self, label="--"))
            self.posCont[0][0].Hide()
            for i in np.arange(codelength):
                self.posCont[0][i+1].Hide()
                self.posCont[0][i+1] = wx.StaticBitmap(self, -1, 
                png, size=(png.GetWidth(), png.GetHeight()))
            for i in np.arange(Ncolors):
                self.posCont[i+1][0].Hide()
                img = wx.Image(path+'img/%d.png' % self.parent.gamePanel.colorIDX[i], 
                        wx.BITMAP_TYPE_ANY).ConvertToBitmap()
                self.posCont[i+1][0] = wx.StaticBitmap(self, -1, 
                img, size=(img.GetWidth(), img.GetHeight()))

            posGrid.AddMany(self.posCont.flatten())
            posSizer.Add(posGrid, proportion=1, 
                flag=wx.ALL | wx.EXPAND, border=5)
            self.SetSizer(posSizer)


    class ColorStats(wx.Panel):
        """ This panel is for the color statistics """ 
        def __init__(self, parent, **kwargs):
            """Constructor"""
            wx.Panel.__init__(self, parent=parent)
            self.parent = parent
            Ncolors = self.parent.gamePanel.game.Ncolors
            codelength = self.parent.gamePanel.game.codelength
            colSizer = wx.BoxSizer(wx.VERTICAL)
            
            font = wx.Font(pointSize=16, weight=wx.BOLD, style=wx.NORMAL,
                family=wx.SYS_SYSTEM_FONT)
            colText = wx.StaticText(self, label="Color count")
            colText.SetFont(font)

            colSizer.Add(colText, flag=wx.LEFT, border=50)

            #-------- pegbox
            colGrid = wx.FlexGridSizer(Ncolors+1, codelength+2, 5, 5)            
            self.colCont = np.empty(shape=(Ncolors+1, codelength+2), 
                dtype=object)
            
            for (i,j),_ in np.ndenumerate(self.colCont):
                self.colCont[i][j] = (wx.StaticText(self, label="--"))
            self.colCont[0][0].Hide()
            for i in np.arange(codelength+1):
                self.colCont[0][i+1].Hide()
                self.colCont[0][i+1] = wx.StaticText(self, label="%d"%i)
            
            for i in np.arange(Ncolors):
                self.colCont[i+1][0].Hide()
                img = wx.Image(path+'img/%d.png' % self.parent.gamePanel.colorIDX[i], 
                        wx.BITMAP_TYPE_ANY).ConvertToBitmap()
                self.colCont[i+1][0] = wx.StaticBitmap(self, -1, 
                img, size=(img.GetWidth(), img.GetHeight()))

            colGrid.AddMany(self.colCont.flatten())
            colSizer.Add(colGrid, proportion=1, 
                flag=wx.ALL | wx.EXPAND, border=5)
            self.SetSizer(colSizer)

    class FeasibleSetStats(wx.Panel):
        """ This panel is for information about the feasible set """ 
        def __init__(self, parent, **kwargs):
            """Constructor"""
            wx.Panel.__init__(self, parent=parent)
            self.parent = parent
            Ncolors = self.parent.gamePanel.game.Ncolors
            codelength = self.parent.gamePanel.game.codelength

            fsSizer = wx.BoxSizer(wx.VERTICAL)
            font = wx.Font(pointSize=16, weight=wx.BOLD, style=wx.NORMAL,
                family=wx.SYS_SYSTEM_FONT)
            fsText = wx.StaticText(self, label="Feasible set")
            fsText.SetFont(font)
            fsSizer.Add(fsText, flag=wx.LEFT, border=50)
            fsSizer.Add((-1, 10))

            sizeSZ = wx.BoxSizer(wx.HORIZONTAL)
            sizeSZ.Add(wx.StaticText(self, label="Size of set:"))
            self.fsSizeTxt = wx.StaticText(self, label="--")
            sizeSZ.Add(self.fsSizeTxt)
            fsSizer.Add(sizeSZ)

            entSZ = wx.BoxSizer(wx.HORIZONTAL)
            entSZ.Add(wx.StaticText(self, label="Shannon entropy of set:"))
            self.entTxt = wx.StaticText(self, label="--")
            entSZ.Add(self.entTxt)
            fsSizer.Add(entSZ)

            fsSizer.Add(wx.StaticText(self, label="Top combinations:"))
            nTopItems = 5
            topGrid = wx.FlexGridSizer(nTopItems, codelength+1, 5, 5)
            self.topCont = np.empty(shape=(nTopItems, codelength+1), 
                dtype=object)
            img = wx.Image(path+'img/0.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            for index, i in np.ndenumerate(self.topCont):
                if index[1] == codelength:
                    self.topCont[index] = wx.StaticText(self, label="--")
                else:
                    self.topCont[index] = wx.StaticBitmap(self, -1, 
                        img, size=(img.GetWidth(), img.GetHeight()))
                self.topCont[index].Hide()

            topGrid.AddMany(self.topCont.flatten())
            fsSizer.Add(topGrid, proportion=1, 
                flag=wx.ALL | wx.EXPAND, border=5)
            self.SetSizer(fsSizer)

    #----------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        """ This is the main CONSTRUCTOR for the stats panel """
        super(StatsApp, self).__init__(
            style=wx.SYSTEM_MENU | wx.CAPTION, 
            title='Game Statistics',
            *args, **kwargs) 
        self.ResetStats()
        self.parent = args[0]

    def Redraw(self):
        """ Needs to be called after panel is updated """
        if self.IsShown():
            self.Hide()
            self.Show()

    def ResetStats(self):
        """ Reset statistics for new game """
        self.step = -1
        if hasattr(self, 'mainSizer'):
            self.mainSizer.Clear(True)
        self.mainSizer = wx.BoxSizer(wx.HORIZONTAL)
        noGameTxt = wx.StaticText(self, label="No stats available. " \
            "Creat new game first.")
        self.mainSizer.Add(noGameTxt, flag=wx.LEFT|wx.TOP, border=10)
        self.mainSizer.Add((-1, 10))
        self.SetSize((850, 400))
        self.SetPosition((460,50))
        self.SetSizer(self.mainSizer)
        self.Layout()

    def InitializeStats(self, panel):
        """ Initialize statistics if first called """
        self.gamePanel = panel
        self.mainSizer.Clear(True)
        self.positionPanel = self.PositionStats(self)
        self.mainSizer.Add(self.positionPanel, 1, 
            wx.LEFT | wx.TOP | wx.BOTTOM | wx.EXPAND, 10)
        self.colorPanel = self.ColorStats(self)
        self.mainSizer.Add(self.colorPanel, 1, 
            wx.LEFT | wx.TOP | wx.BOTTOM | wx.EXPAND, 10)
        self.setPanel = self.FeasibleSetStats(self)
        self.mainSizer.Add(self.setPanel, 1, 
            wx.LEFT | wx.TOP | wx.BOTTOM | wx.RIGHT | wx.EXPAND, 10)
        self.Layout()
        
    def UpdateStats(self, pos, col, misc):
        """ Update statistics as the game unfolds """
        #--------- update position stats
        itidx = self.positionPanel.posCont.shape
        for i in np.arange(itidx[0]-1):
            for j in np.arange(itidx[1]-1):
                if pos[i][j] == 0: label = '--'
                else: label = '%.2f' % pos[i][j]         
                self.positionPanel.posCont[i+1][j+1].SetLabel(label)
        #--------- update color stats
        itidx = self.colorPanel.colCont.shape
        for i in np.arange(itidx[0]-1):
            for j in np.arange(itidx[1]-1):
                if col[i][j] == 0: label = '--'
                else: label = '%.2f' % col[i][j] 
                self.colorPanel.colCont[i+1][j+1].SetLabel(label)
        #--------- update fs stats
        self.setPanel.fsSizeTxt.SetLabel(str(misc[0]))
        self.setPanel.entTxt.SetLabel('%.2f' % misc[1])
        itidx = self.setPanel.topCont.shape
        maxidx = misc[2].shape
        for i in np.arange(itidx[0]):
            for j in np.arange(itidx[1]-1):
                self.setPanel.topCont[i][j].Hide()
                if i < maxidx[0]:
                    img = wx.Image(path+'img/%d.png' % 
                        self.gamePanel.colorIDX[misc[2][i][j]-1], 
                        wx.BITMAP_TYPE_ANY).ConvertToBitmap()
                    self.setPanel.topCont[i][j].SetBitmap(img)
                    self.setPanel.topCont[i][j].Show()
            self.setPanel.topCont[i][-1].Hide()
            if i < maxidx[0]:
                self.setPanel.topCont[i][-1].SetLabel("%.2f" % misc[3][i])
                self.setPanel.topCont[i][-1].Show()
        self.Layout()



########################################################################
class MMindApp(wx.Frame):
    """ 
        Main application class controlling creating and 
        switching between the different panels (above) 
    """
    #----------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        """ CONSTRUCTOR """
        super(MMindApp, self).__init__(
            style=wx.SYSTEM_MENU | wx.CAPTION |  wx.CLOSE_BOX, 
            title='Mastermind',
            *args, **kwargs) 
        
        self.SetSize((400, 600))
        self.SetPosition((50,50))

        #----------- top toolbar
        self.toolbar = self.CreateToolBar(style=wx.TB_TEXT)
        newtool = self.toolbar.AddLabelTool( wx.ID_ANY, 'New', 
            wx.Bitmap(path+'icons/stock_new.png'))
        statstool = self.toolbar.AddLabelTool( wx.ID_ANY, 'Stats', 
            wx.Bitmap(path+'icons/stock_save.png'))
        quittool = self.toolbar.AddLabelTool( wx.ID_ANY, 'Quit', 
            wx.Bitmap(path+'icons/stock_exit.png'))
        self.toolbar.Realize()
        self.Bind(wx.EVT_TOOL, self.loadNewGamePanel, newtool)  
        self.Bind(wx.EVT_TOOL, self.toggleStats, statstool)
        self.Bind(wx.EVT_TOOL, self.OnQuit, quittool)

        #----------- create panels
        self.start_panel = StartPanel(self)
        self.new_game_panel = NewGame(self)
        self.new_game_panel.Hide()
        self.activePanel = self.start_panel
        self.statsFrame = StatsApp(self)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.start_panel, 1, wx.EXPAND)
        self.sizer.Add(self.new_game_panel, 1, wx.EXPAND)

        self.SetSizer(self.sizer)
        self.Show(True)

    #----------------------------------------------------------------------
    def toggleStats(self, event):
        """ toggle the statistics window """
        if self.statsFrame.IsShown() or self.statsFrame.IsShownOnScreen():
            self.statsFrame.Hide()
        else:
            self.statsFrame.Show()
            if self.activePanel == self.play_game_panel:
                self.play_game_panel.UpdateStatistics()

    def loadNewGamePanel(self, event):
        """ open create new game panel """
        self.activePanel.Hide()
        self.activePanel = self.new_game_panel
        self.new_game_panel.Show()
        self.statsFrame.ResetStats()
        self.Layout()

    def createNewGame(self, event):
        """ 
            executed once player clicked on 'create game' 
            used to pass on data to mastermind.py 
        """
        code_jar = [int(tc.GetValue())
            for tc in self.new_game_panel.pegtc]
        codelength = self.new_game_panel.clengthfield.GetValue()
        print [btn.GetValue() for btn in 
            self.new_game_panel.rbtns]

        itemindex = np.where(np.array([btn.GetValue() for btn in 
            self.new_game_panel.rbtns]) == True)[0][0]

        if hasattr(self, 'play_game_panel'):
            self.play_game_panel.Destroy()

        self.play_game_panel = PlayGame(self, mode=itemindex,
            t=self.new_game_panel.degreeIpt.GetValue(), 
            r=self.new_game_panel.orderIpt.GetValue(),
            p=self.new_game_panel.mixIpt.GetValue(),
            codejar=code_jar, codelength=codelength, logging=True)
        self.activePanel.Hide()
        self.activePanel = self.play_game_panel
        self.play_game_panel.Show()
        self.sizer.Add(self.play_game_panel, 1, wx.EXPAND)
        self.Layout()

    def OnQuit(self, e):
        """ close on quit """
        self.Close()


""" If executed as main, run the following code: """
if __name__ == "__main__":
    app = wx.App()
    MMindApp(None)
    app.MainLoop()