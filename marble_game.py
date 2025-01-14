import os, yaml, argparse
import numpy as np
from scipy.interpolate import PchipInterpolator as pci
from psychopy import core, event, visual
from psychopy.iohub.client import launchHubServer
from psychopy.visual.windowwarp import Warper
from keyboard import *

parser = argparse.ArgumentParser(description='Marble game parameters')
parser.add_argument('-c','--config', help='Configuration file',default='demo')
parser.add_argument('-fs','--fullscreen', help='Fullscreen mode', action='store_true', default=False)
parser.add_argument('-p','--perspective', help='Perspective mode', action='store_false', default=True)
args = parser.parse_args()


def gen_trough_shape(win, full_angle_deg=60, width=0.8, edge_width=0.04,
        num_pts=30, line_width=5, line_color=[0.6,0.6,0.6],
        xpos=0, ypos=0):
    half_trough_angle = np.deg2rad(0.5*full_angle_deg)
    circ_points = 3/2*np.pi+np.linspace(-half_trough_angle,half_trough_angle,num_pts)
    circle_rad = 0.5*width/np.sin(half_trough_angle)
    xs = circle_rad*np.cos(circ_points)
    ys = circle_rad*np.sin(circ_points)+circle_rad

    xs = np.concatenate([[xs[0]-edge_width],xs,[xs[-1]+edge_width]])
    ys = np.concatenate([[ys[0]],ys,[ys[-1]]])

    vertices = np.vstack([xs,ys]).T
    shape = visual.ShapeStim(win,
        vertices=vertices,
        lineWidth=line_width,
        lineColor=line_color,
        closeShape=False,
        pos=(xpos,ypos), interpolate=True)
    return shape

def gen_semicirc_shape(win, radius=0.4,
        num_pts=30, line_width=1.5, line_color=[0.6,0.6,0.6],
        xpos=0, ypos=0):
    circ_points = np.linspace(np.pi,2*np.pi,num_pts)
    xs = radius*np.cos(circ_points)
    ys = radius*np.sin(circ_points)

    vertices = np.vstack([xs,ys]).T
    shape = visual.ShapeStim(win,
        vertices=vertices,
        lineWidth=line_width,
        lineColor=line_color,
        closeShape=False,
        pos=(xpos,ypos), interpolate=True)
    return shape

def gen_shadow_shape(win, radius=0.4,
        shadow_offset=0.1, shadow_xscale=0.8, shadow_yscale=1.5, opacity=0.5,
        num_pts=30, line_width=1.5,
        fill_color=[0.6,0.6,0.6], line_color=[0.6,0.6,0.6],
        xpos=0, ypos=0):
    circ_points = np.linspace(0,2*np.pi,num_pts)
    xs = shadow_xscale*radius*np.cos(circ_points)
    ys = shadow_offset+shadow_yscale*radius*np.sin(circ_points)

    vertices = np.vstack([xs,ys]).T
    shape = visual.ShapeStim(win,
        vertices=vertices,
        lineWidth=line_width,
        lineColor=line_color,
        fillColor=fill_color,
        opacity=opacity,
        pos=(xpos,ypos), interpolate=True)
    return shape

def gen_course_path(targets=[0,0,25,-35,15,-15,0,0],
        times=[0,0.5,2,3.5,5,6.5,8,8.5], speed=0.25, step_size=0.01):
    ypos = speed*np.array(times)
    course_interp = pci(ypos,targets)
    ypos_out = np.arange(ypos[0],ypos[-1]+step_size,step_size)
    angle_out = course_interp(ypos_out)
    return ypos_out, angle_out

def gen_course_shape(win, course_y_raw, course_angle_raw,
        course_rad, course_angle_width=15,
        line_width=1.5, fill_color=[0.6,0.6,0.6], line_color=[0.6,0.6,0.6],
        xpos=0, ypos=-0.35, endcap_points=10):
    course_angles_lh = np.deg2rad(course_angle_raw-0.5*course_angle_width)
    course_xs_lh = course_rad*np.sin(course_angles_lh)
    course_angles_rh = np.deg2rad(course_angle_raw+0.5*course_angle_width)
    course_xs_rh = course_rad*np.sin(course_angles_rh)
    endcap_angles_bottom = np.linspace(course_angles_rh[0],course_angles_lh[0],endcap_points)[1:-1]
    endcap_angles_top = np.linspace(course_angles_lh[-1],course_angles_rh[-1],endcap_points)[1:-1]
    endcap_xs_bottom = course_rad*np.sin(endcap_angles_bottom)
    endcap_xs_top = course_rad*np.sin(endcap_angles_top)

    endcap_ys_bottom = np.linspace(course_y_raw[0],course_y_raw[0],endcap_points)[1:-1]
    endcap_ys_top = np.linspace(course_y_raw[-1],course_y_raw[-1],endcap_points)[1:-1]

    xs = np.concatenate([endcap_xs_bottom,course_xs_lh,endcap_xs_top,course_xs_rh[::-1]])
    ys = np.concatenate([endcap_ys_bottom,course_y_raw,endcap_ys_top,course_y_raw[::-1]])
    ys += course_rad*(1-np.cos(np.concatenate([endcap_angles_bottom,course_angles_lh,
        endcap_angles_top,course_angles_rh[::-1]])))
    vertices = np.vstack([xs,ys]).T
    shape = visual.ShapeStim(win,
        vertices=vertices,
        lineWidth=line_width,
        lineColor=line_color,
        fillColor=fill_color,
        pos=(xpos,ypos), interpolate=True)
    return shape

class MarbleGame:

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

        self.perspective_on = self.args.perspective

        self.win = visual.Window(size=(self.config['screen_width'], self.config['screen_height']),
                     color=self.config['bg_color'], units='height',
                     fullscr=self.args.fullscreen, useFBO=self.perspective_on)

        self.kb = KeyboardWrapper()

        self.warper = Warper(self.win,
            warp='warpfile',
            warpfile = 'config/perspective.data')

        # add key controls
        event.globalKeys.add(key='q', modifiers=['alt'], func=self.quit)
        event.globalKeys.add(key='r', func=self.reset_course)

        # basic graphics and colors
        self.cue_color = self.config['cue_color']
        self.trough_width = self.config['trough_width']
        self.trough_edge_width = self.config['trough_edge_width']
        self.trough_full_angle = self.config['trough_full_angle']
        self.num_troughs = 8
        self.trough_speed = self.config['trough_speed']
        self.trough_color = self.config['trough_color']
        self.trough_line_color = self.config['trough_line_color']
        self.trough_edge_color = self.config['trough_edge_color']
        self.course_color = self.config['course_color']
        self.troughs = [gen_trough_shape(self.win, full_angle_deg=self.trough_full_angle,
            width=self.trough_width, edge_width=self.trough_edge_width,
            line_color=self.trough_line_color) for t in range(self.num_troughs)]
        self.troughs_ypos = np.linspace(1,-1,self.num_troughs)
        self.bg_grating = visual.GratingStim(self.win, texRes=1024,
            color=self.trough_color,size=(self.trough_width,1.0),contrast=0.25)
        self.lh_trough_rect = visual.Rect(self.win, width=self.trough_edge_width, height=1,
            pos=(0.5*(self.trough_width+self.trough_edge_width),0),color=self.trough_edge_color)
        self.rh_trough_rect = visual.Rect(self.win, width=self.trough_edge_width, height=1,
            pos=(-0.5*(self.trough_width+self.trough_edge_width),0),color=self.trough_edge_color)

        # marble logic
        self.marble_base_ypos = self.config['marble_base_ypos']
        self.marble_color = self.config['marble_color']
        self.marble_border_color = self.config['marble_border_color']
        self.marble_shadow_color = self.config['marble_shadow_color']
        self.marble_rad = self.config['marble_rad']
        self.marble_circ = 2*np.pi*self.marble_rad
        self.marble_yscales = [1]
        self.marble = visual.Circle(self.win, radius=self.marble_rad, fillColor=self.marble_color,
            lineColor=self.marble_border_color, interpolate=True, pos=(0,self.marble_base_ypos))
        self.marble_semicircs = [gen_semicirc_shape(self.win, radius=self.marble_rad,
            num_pts=20, line_width=1.5, line_color = self.marble_border_color,
            xpos=0, ypos=self.marble_base_ypos) for marble in self.marble_yscales]
        self.marble_shadow = gen_shadow_shape(self.win, radius=self.marble_rad,
            shadow_offset=-0.4*self.marble_rad, shadow_xscale=0.9, shadow_yscale=0.9, opacity=0.25,
            num_pts=30, line_width=1.0,
            fill_color=self.marble_shadow_color, line_color=self.marble_shadow_color,
            xpos=0, ypos=self.marble_base_ypos)
        self.marble_neutral_angle = 0
        self.kb_neutral_angle = self.kb.config['neutral_angle']
        self.marble_trough_rad = 0.5*self.trough_width/np.sin(np.deg2rad(0.5*self.trough_full_angle))

        self.marble_angle = self.marble_neutral_angle
        self.marble_velocity = 0
        self.marble_xpos = 0
        self.marble_ypos = self.marble_base_ypos
        self.angle_gain = self.config['kb_angle_gain']
        self.marble_rota_coef = self.config['marble_rota_coef']

        # course example
        self.target_time_spacing = 1.0 #1.5
        self.start_end_time_spacing = 0.5
        # self.targets = [15,-35,-15,-25,30,15,5,-20]
        self.targets = [25,-35,15,-15,30]
        self.target_times = np.arange(self.target_time_spacing,
            self.target_time_spacing*(len(self.targets)+1),
            self.target_time_spacing)
        self.course_start_end_targets = [0,0]
        self.course_start_times = [0,self.start_end_time_spacing]
        self.course_end_times = [self.target_time_spacing,self.target_time_spacing+self.start_end_time_spacing]
        self.course_targets = np.concatenate([self.course_start_end_targets,
            self.targets,self.course_start_end_targets])
        self.course_times = np.concatenate([self.course_start_times,
            self.course_start_times[1]+np.array(self.target_times),
            self.target_times[-1]+np.array(self.course_end_times)])
        self.course_y_raw, self.course_angle_raw = gen_course_path(self.course_targets,
            self.course_times, self.trough_speed)
        self.course_example = gen_course_shape(self.win,
            self.course_y_raw, self.course_angle_raw,
            self.marble_trough_rad,
            fill_color=self.course_color,
            line_color=self.course_color)

        # timing check
        self.clock = core.Clock()
        self.last_time = 0.0
        self.frame_time = 0.0
        self.frame_history = np.zeros(100)
        self.frame_msg = visual.TextBox2(win=self.win,
            text='16.6', pos=(-0.4,0.4),
            color=self.cue_color, letterHeight=0.08,
            units='height', autoDraw=False)#, autoDraw=True)

        # self.debug_msg = visual.TextBox2(win=self.win,
        #     text='', pos=(-0.4,0.3),
        #     color=self.cue_color, letterHeight=0.08,
        #     units='height',
        #     autoDraw=True)

        # keyboard setup
        self.kb.send_command('mode_action_mirror_rh')
        self.display_hand = 'lh' # lh or rh

        self.game_running = True

    def reset_course(self):
        self.course_example.pos = (0,0.5)

    def update_frame_time(self):
        self.frame_time = self.clock.getTime()-self.last_time
        self.last_time = self.clock.getTime()
        self.frame_history = np.roll(self.frame_history,1)
        self.frame_history[0] = self.frame_time
        self.frame_msg.text = str(np.round(1000*np.mean(self.frame_history),1))

    def update_troughs(self):
        self.course_example.pos = (0, self.course_example.pos[1]-self.frame_time*self.trough_speed)
        self.troughs_ypos -= self.frame_time*self.trough_speed
        for idx, trough in enumerate(self.troughs):
            if self.troughs_ypos[idx] < -1:
                self.troughs_ypos[idx] += 2
            trough.pos = (0, self.troughs_ypos[idx])

    def update_marble(self):
        # update angular position
        all_angle_raw = self.kb.all_pos[:]
        all_vel_raw = self.kb.all_vel[:]
        if self.display_hand == 'lh':
            motor_idx = 1
        else:
            motor_idx = 0
        self.marble_angle = (all_angle_raw[motor_idx]-self.kb_neutral_angle)*self.angle_gain
        self.marble_velocity = all_vel_raw[motor_idx]*self.angle_gain

        if self.display_hand == 'lh':
            self.marble_angle = -self.marble_angle
            self.marble_velocity = -self.marble_velocity

        self.marble_xpos = self.marble_trough_rad*np.sin(np.deg2rad(self.marble_angle))
        self.marble_ypos = self.marble_base_ypos+self.marble_trough_rad*(1-np.cos(np.deg2rad(self.marble_angle)))
        self.marble.pos = (self.marble_xpos,self.marble_ypos)

        # update marble shadow
        self.marble_shadow.pos = (self.marble_xpos,self.marble_ypos)
        self.marble_shadow.size = (1,1+(1-np.cos(np.deg2rad(self.marble_angle))))
        self.marble_shadow.ori = 0.4*self.marble_angle

        # update rolling semicircs
        y_distance_travelled = self.trough_speed*self.frame_time
        yscale_travel_rough = y_distance_travelled/(0.4*self.marble_circ)
        x_distance_travelled = 2*self.marble_trough_rad*np.pi*self.marble_velocity/360*self.frame_time
        xscale_travel_rough = x_distance_travelled/(0.4*self.marble_circ)
        for idx, marble_semicirc in enumerate(self.marble_semicircs):
            # self.marble_yscales[idx] -= yscale_travel_rough
            self.marble_yscales[idx] -= np.sqrt(yscale_travel_rough**2+xscale_travel_rough**2)
            if self.marble_yscales[idx] < -1:
                self.marble_yscales[idx] += 2
            marble_semicirc.size = (1,np.sin(np.pi/2-(self.marble_yscales[idx]-1)*np.pi/2))
            marble_semicirc.pos = (self.marble_xpos,self.marble_ypos)
            marble_semicirc.ori = self.marble_rota_coef*self.marble_velocity

    # main event loop
    def run_main_loop(self):
        while self.game_running:
            self.update_frame_time()
            self.bg_grating.draw()
            self.lh_trough_rect.draw()
            self.rh_trough_rect.draw()
            self.update_troughs()
            for trough in self.troughs:
                trough.draw()
            self.course_example.draw()
            self.update_marble()
            self.marble_shadow.draw()
            self.marble.draw()
            for marble_semicirc in self.marble_semicircs:
                marble_semicirc.draw()
            self.win.flip()

    def quit(self):
        self.game_running = False
        core.quit()

if __name__ == '__main__':
    game = MarbleGame()
    game.run_main_loop()
