import os, yaml, argparse
import numpy as np
from psychopy import core, event, visual
from psychopy.iohub.client import launchHubServer

parser = argparse.ArgumentParser(description='Marble game parameters')
parser.add_argument('-c','--config', help='Configuration file',default='wedge_demo')
parser.add_argument('-fs','--fullscreen', help='Fullscreen mode', action='store_true', default=False)
args = parser.parse_args()

def gen_key_shape(win, width=0.1, height=0.05,
        corner_rad=0.02, line_width=2.5, corner_pts=5,
        fill_color=[-0.5,-0.5,-0.5], line_color=[0.1,0.1,0.1],
        xpos=0, ypos=0):
    num_circ_points = 4*corner_pts+5
    circ_points = np.linspace(0,2*np.pi,num=num_circ_points)
    cxs = corner_rad*np.cos(circ_points)
    cys = corner_rad*np.sin(circ_points)
    qts = corner_pts+2
    xs = np.zeros(num_circ_points+3)
    xs[0:qts] = cxs[0:qts]+0.5*width
    xs[qts:2*qts] = cxs[qts-1:2*qts-1]-0.5*width
    xs[2*qts:3*qts] = cxs[2*qts-2:3*qts-2]-0.5*width
    xs[3*qts:4*qts] = cxs[3*qts-3:4*qts-3]+0.5*width
    ys = np.zeros(num_circ_points+3)
    ys[0:qts] = cys[0:qts]+0.5*height
    ys[qts:2*qts] = cys[qts-1:2*qts-1]+0.5*height
    ys[2*qts:3*qts] = cys[2*qts-2:3*qts-2]-0.5*height
    ys[3*qts:4*qts] = cys[3*qts-3:4*qts-3]-0.5*height
    vertices = np.vstack([xs,ys]).T
    shape = visual.ShapeStim(win,
        vertices=vertices,
        lineWidth=line_width,
        lineColor=line_color,
        fillColor=fill_color,
        pos=(xpos,ypos), interpolate=True)
    return shape

def gen_wedge_shape(win, task_rad=1.0, width=0.1, height=0.05,
        corner_rad=0.02, line_width=2.5, corner_pts=5,
        fill_color=[-0.5,-0.5,-0.5], line_color=[0.1,0.1,0.1],
        xpos=0, ypos=0):
    num_circ_points = 4*corner_pts+5
    circ_points = np.linspace(0,2*np.pi,num=num_circ_points)
    cxs = corner_rad*np.cos(circ_points)
    cys = corner_rad*np.sin(circ_points)
    qts = corner_pts+2
    xs = np.zeros(num_circ_points+3)
    xs[0:qts] = cxs[0:qts]+0.5*width
    xs[qts:2*qts] = cxs[qts-1:2*qts-1]-0.5*width
    xs[2*qts:3*qts] = cxs[2*qts-2:3*qts-2]-0.5*width
    xs[3*qts:4*qts] = cxs[3*qts-3:4*qts-3]+0.5*width
    ys = np.zeros(num_circ_points+3)
    ys[0:qts] = cys[0:qts]+0.5*height
    ys[qts:2*qts] = cys[qts-1:2*qts-1]+0.5*height
    ys[2*qts:3*qts] = cys[2*qts-2:3*qts-2]-0.5*height
    ys[3*qts:4*qts] = cys[3*qts-3:4*qts-3]-0.5*height
    vertices = np.vstack([xs*(ys/task_rad+1),ys]).T
    shape = visual.ShapeStim(win,
        vertices=vertices,
        lineWidth=line_width,
        lineColor=line_color,
        fillColor=fill_color,
        pos=(xpos,ypos), interpolate=True)
    return shape

class WedgeGame:

    def __init__(self):
        # load command line args
        self.args = args

        # load config
        self.config_dir = os.path.join('config',self.args.config+'.yml')
        try:
            with open(self.config_dir) as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            print('Configuration file '+self.args.config+'.yml not found')
            sys.exit(1)

        self.win = visual.Window(size=(self.config['screen_width'], self.config['screen_height']),
                     color=self.config['bg_color'], units='height',
                     fullscr=self.args.fullscreen)

        # add key controls
        event.globalKeys.add(key='q', modifiers=['alt'], func=self.quit)
        self.io = launchHubServer()
        self.kb = self.io.devices.keyboard
        self.left_key = ['a']
        self.right_key = ['d']
        self.key_codes = self.left_key+self.right_key
        self.left_keydown = False
        self.right_keydown = False

        # graphics and colors
        self.cue_color = self.config['cue_color']
        self.target_color = self.config['target_color']

        self.task_center_y = 0
        self.task_rad = self.config['task_radius'] # screen heights
        self.task_ori = 0 # straight up

        self.wedge_height = self.config['wedge_height']
        self.wedge_size = self.config['wedge_size']
        self.wedge_width = (2*self.task_rad
            *np.sin(np.deg2rad(self.wedge_size)/2))
        self.wedge_corner_rad = self.config['wedge_corner_rad']

        self.wedge = gen_wedge_shape(self.win, self.task_rad,
            width=self.wedge_width-2*self.wedge_corner_rad,
            height=self.wedge_height-2*self.wedge_corner_rad,
            corner_rad=self.wedge_corner_rad, line_width=1.0,
            fill_color=self.cue_color, line_color=self.cue_color)

        self.target_height = self.config['target_height']
        self.target_size = self.config['target_size']
        self.target_width = (2*self.task_rad
            *np.sin(np.deg2rad(self.target_size+self.wedge_size)/2))
        self.target_corner_rad = self.config['target_corner_rad']

        self.target_oris = self.config['target_pos']
        self.num_targets = len(self.target_oris)

        self.targets = [gen_wedge_shape(self.win, self.task_rad,
            width=self.target_width-2*self.target_corner_rad,
            height=self.target_height-2*self.target_corner_rad,
            corner_rad=self.target_corner_rad, line_width=1.0,
            fill_color=self.target_color, line_color=self.target_color)
                for target in range(self.num_targets)]

        for target_idx, target in enumerate(self.targets):
            target.ori = self.target_oris[target_idx]
            task_ori_rad = np.deg2rad(self.target_oris[target_idx])
            ypos = -self.task_rad + self.task_rad*np.cos(task_ori_rad)
            xpos = self.task_rad*np.sin(task_ori_rad)
            target.pos = (xpos,ypos)

        # game logic
        self.game_running = True
        self.input_direction = 0

        # timing
        self.clock = core.Clock()
        self.last_time = 0.0
        self.frame_time = 0.0
        self.rotation_speed = 60.0 # deg/sec

    def check_keys(self):
        events = self.kb.getKeys()
        for kbe in events:
            if kbe.key in self.key_codes:
                if kbe.type == 'KEYBOARD_PRESS':
                    if kbe.key in self.left_key:
                        self.left_keydown = True
                    elif kbe.key in self.right_key:
                        self.right_keydown = True
                elif kbe.type == 'KEYBOARD_RELEASE':
                    if kbe.key in self.left_key:
                        self.left_keydown = False
                    elif kbe.key in self.right_key:
                        self.right_keydown = False
        if self.left_keydown:
            if not(self.right_keydown):
                self.input_direction = -1
            else:
                self.input_direction = 0
        elif self.right_keydown:
            if not(self.left_keydown):
                self.input_direction = 1
            else:
                self.input_direction = 0
        else:
            self.input_direction = 0

    def update_frame_time(self):
        self.frame_time = self.clock.getTime()-self.last_time
        self.last_time = self.clock.getTime()

    def update_wedge(self):
        self.task_ori += self.rotation_speed*self.input_direction*self.frame_time

        task_ori_rad = np.deg2rad(self.task_ori)

        task_ypos = -self.task_rad + self.task_rad*np.cos(task_ori_rad)
        task_xpos = self.task_rad*np.sin(task_ori_rad)

        self.wedge.ori = self.task_ori
        self.wedge.pos = (task_xpos, task_ypos)

    def run_main_loop(self):
        while self.game_running:
            self.update_frame_time()
            self.check_keys()
            self.update_wedge()
            for target in self.targets:
                target.draw()
            self.wedge.draw()
            self.win.flip()

    def quit(self):
        self.game_running = False
        core.quit()

if __name__ == '__main__':
    game = WedgeGame()
    game.run_main_loop()
