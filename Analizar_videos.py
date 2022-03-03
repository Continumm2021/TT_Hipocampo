from logging import error, warning
from unittest.loader import VALID_MODULE_NAME
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#from shapely.geometry import Poligon
from scipy.interpolate import make_interp_spline, BSpline

#paameters to assamble the directory of the videos
main_folder = 'D:\\Vida\\Trabajo de titulo\\Python_code\\'
vid_names = ["d4_perfil_S_liso","d4_perfil_S_rugoso","d3_perfil_S",
            "d2_perfil_S","cola_S_frontal","cola_S_lateral", "d4_4f_2f",
            "d4_S_ortogonal_frente","d4_S_ortogonal_lateral", "d4_diagonal",
             "d4_diagonal_n", "d4_4f_2f"]
#directory of a document with the vacuum measurement of each video. The headers of the document must have the same name as the folder
vacuum_file_name = "presiones_tesis.csv"
vacuum_dir = main_folder + vacuum_file_name


#Calculates the coordintares of the middle point
def calc_half(coord1, coord2):
    x1 = coord1[0]
    y1 = coord1[1]
    x2 = coord1[0]
    y2 = coord1[1]
    xm = (x2-x1)/2
    ym = (y2-y1)/2
    half_coord = (xm,ym)
    return half_coord

#Calculate the distance between 2 points
def calc_distance(coord1, coord2):
    x1 = coord1[0]
    y1 = coord1[1]
    x2 = coord2[0]
    y2 = coord2[1]
    d = np.sqrt((x2-x1)**2+(y2-y1)**2)
    return d

#3rd degree polinome
def fit_func_poli(x,a,b,c,d):
    return a + (b*x) + (c*x**2) + (d*x**3)

#first derivative
def fit_func_poli_deriv(x,b,c,d):
    return b + (2*c*x) + (3*d*x**2)

#second derivative
def fit_func_poli_2deriv(x,c,d):
    return 2*c + 6*d*x

#sine equation
def fit_func_sin(x, freq, amplitude, phase, offset):
    return np.sin(x * freq + phase) * amplitude + offset

#first derivative
def fit_func_sin_deriv(x, freq, amplitude, phase):
    return -freq*amplitude*np.cos(x*freq+phase)

#second derivative
def fit_func_sin_2deriv(x, freq, amplitude, phase):
    return -(freq**2)*amplitude*np.sin(x*freq+phase)

#Calculate the theoretical curvature
def calc_curvature_t(x,b,c,d):
    numer = np.abs(fit_func_poli_2deriv(x,c,d))
    denom = (1+(fit_func_poli_deriv(x,b,c,d))**2)**(3/2)
    return numer/denom

#calculate the theoretical deflection using design parameters to estimate the curvature radius angle
def calc_deflection_t(R,side_l,shortening):
    #print(np.pi,side_l,shortening)
    beta = np.pi - 2 * np.arctan(2*side_l/shortening)
    deflect = R * (1-np.cos(beta/2)) 
    return deflect

#Return y value calculating the slope and intercept using 2 set of coordinates as reference
def point_slope(coord1, coord2):
    x1 = coord1[0]
    y1 = coord1[1]
    x2 = coord2[0]
    y2 = coord2[1]
    m, b = np.polyfit([x1,x2],[y1,y2],1)
    return m, b

def analize_deflection_t(x,y,a,b,c,d):
    #Designs parameters passed as global variables
    l = 30.62
    dh = 2.12*4
    deflections_t = []
    x_tm = []
    y_tm = []
    m1, b1 = point_slope((x[0],y[0]), (x[-1],y[-1]))
    #uses only the lines perpendicular to the line from the first and last data
    m2 = -1/m1
    for i in range(1, len(x)-1,1):
        y_t = y[i]
        x_t = a + b*y_t + c*y_t**2 + d*y_t**3
        b_t = y_t - x_t*m2
        x_tm.append((b_t-b1)/(m1-m2))
        y_tm.append(m1*x_tm[-1]+b1)
        d_t = calc_distance((x_t,y_t),(x_tm[-1],y_tm[-1]))
        side_t = np.sign(x_tm[-1]-x_t)
        deflections_t.append(side_t*d_t)

    index_t_der = deflections_t.index(min(deflections_t))
    index_t_izq = deflections_t.index(max(deflections_t))
    deflect_t_der = deflections_t[index_t_der]
    deflect_t_izq = deflections_t[index_t_izq]

    x_td = a + b*y[index_t_der+1] + c*y[index_t_der+1]**2 + d*y[index_t_der+1]**3
    y_td = y[index_t_der+1]
    x_ti = a + b*y[index_t_izq+1] + c*y[index_t_izq+1]**2 + d*y[index_t_izq+1]**3
    y_ti = y[index_t_izq+1]
    coord_ti = (x_ti,y_ti)
    coord_td = (x_td,y_td)
    coord_tmi = (x_tm[index_t_izq],y_tm[index_t_izq])
    coord_tmd = (x_tm[index_t_der],y_tm[index_t_der])
    curv_radius_ci = 1 / calc_curvature_t(y_ti,b,c,d)
    curv_radius_cd = 1 / calc_curvature_t(y_td,b,c,d)
    deflect_c_izq = calc_deflection_t(curv_radius_ci,l,dh)
    deflect_c_der = calc_deflection_t(curv_radius_cd,l,dh)
    deflect_c = (deflect_c_der,deflect_c_izq)
    deflect_t = (-deflect_t_der,deflect_t_izq)
    print(deflect_c, deflect_t)
    return deflect_c, deflect_t, coord_td, coord_ti, coord_tmd, coord_tmi

def analize_deflection_e(x,y):
    deflections_e = []
    x_em = []
    y_em = []
    m1, b1 = point_slope((x[0],y[0]), (x[-1],y[-1]))
    #uses only the lines perpendicular to the line from the first and last data
    m2 = -1/m1
    theta_inclination = np.arctan(m2) * 180 / np.pi
    for i in range(1, len(x)-1,1):
        b_e = y[i] - x[i]*m2
        x_em.append((b_e-b1)/(m1-m2))
        y_em.append(m1*x_em[-1]+b1)
        d_e = calc_distance((x[i],y[i]),(x_em[-1],y_em[-1]))
        side_e = np.sign(x_em[-1]-x[i])
        deflections_e.append(side_e*d_e)

    index_e_der = deflections_e.index(min(deflections_e))
    index_e_izq = deflections_e.index(max(deflections_e))
    deflect_e_der = deflections_e[index_e_der]
    deflect_e_izq = deflections_e[index_e_izq]
    coord_emi = (x_em[index_e_izq],y_em[index_e_izq])
    coord_ei = (x[index_e_izq+1], y[index_e_izq+1])
    coord_emd = (x_em[index_e_der],y_em[index_e_der])
    coord_ed = (x[index_e_der+1], y[index_e_der+1])

    deflect_e = (-deflect_e_der,deflect_e_izq)
    return deflect_e, coord_emd, coord_emi, coord_ed, coord_ei, theta_inclination
    



#given the directory sorts the data and return it
def handle_dataset(main_dir, vid_name, ds, file_name):
    coord_dict = {}
    file_loc = main_dir + vid_name + file_name + str(ds) + ".csv"
    data = pd.read_csv(file_loc, header=None)
    data = data.reset_index()

    for index, row in data.iterrows():
        coord_dict[index] = (row[0],(1080*ds)-row[1])
    x, y = zip(*coord_dict.values())
    order = np.argsort(y)
    x_n = np.array(x)[order]
    y_n = np.array(y)[order]

    return x_n, y_n

def calc_area(coords):
    polygon = Polygon(coords)
    return polygon.area

#given the data and fitted data calculate the residual squared sum and error
def calc_residual_stats(x,fitted_x):
    residuals = []
    sqrd_residuals = []

    for i in range(len(x)):
        residuals.append(x[i]-fitted_x[i])
        sqrd_residuals.append((x[i]-fitted_x[i])**2)
    RSS = sum(sqrd_residuals)
    RSE = np.sqrt(RSS/(len(x)-2))
    print("RSS:", RSS, "RSE:", RSE)

    return RSS, RSE

#Finds a polinome parameters that fits the data
def fit_to_poli(x,y):
        popt, pcov = curve_fit(fit_func_poli, y, x)

        fitted_x = []
        for item in y:
            fitted_x.append(fit_func_poli(item, *popt))
        
        return fitted_x, popt, pcov

#Finds a sine parameters that fits the data
def fit_to_sine(x,y):
        #initial guesses
        initial_freq = 0.025
        initial_amplitude = 3*np.std(x)/(2**0.5)
        initial_phase = 0.025
        initial_offset = np.mean(x)
        p0=[initial_freq, initial_amplitude,initial_phase, initial_offset]
        popt, pcov = curve_fit(fit_func_sin, y, x, p0=p0)

        fitted_x = []
        for item in y:
            fitted_x.append(fit_func_poli(item, *popt))
        
        return fitted_x, popt, pcov

#Calculate the closest shape of the apendix using a 3rd degree polinome
def graph_dataset(main_dir, vid_name, ds, file_name,x,fitted_x,y,popt,coord_td, coord_ti, coord_tmd, coord_tmi, coord_emd, coord_emi, coord_ed, coord_ei):
        #print(x.shape )
        #xnew = np.linspace(min(np.array(x)), max(np.array(x)), 300)
        #spl = make_interp_spline(np.array(x), np.array(y), k=3)  # type: BSpline
        #power_smooth = spl(xnew)
        file_path = main_dir + vid_name + file_name + str(ds) + ".png"
        plt.rc('axes', titlesize=36)     # fontsize of the axes title
        plt.rc('axes', labelsize=32)    # fontsize of the x and y labels

        fig, ax1 = plt.subplots(1,1)
        fig.set_size_inches(19.2,10.8)
        ax1.set_title("Discretización Eje Neutro del Apéndice")
        ax1.scatter(x,y,color="blue", alpha=0.3)
        ax1.plot(x,y,color="blue")
        ax1.plot((coord_ed[0],coord_emd[0]),(coord_ed[1],coord_emd[1]),color="blue", alpha=0.8)
        ax1.plot((coord_ei[0],coord_emi[0]),(coord_ei[1],coord_emi[1]),color="blue", alpha=0.8, label = "Curva discreta")
        ax1.plot((x[0],x[-1]),(y[0],y[-1]),color="c", label = "Línea media")
        ax1.plot((coord_td[0],coord_tmd[0]),(coord_td[1],coord_tmd[1]),color="green", alpha=0.8)
        ax1.plot((coord_ti[0],coord_tmi[0]),(coord_ti[1],coord_tmi[1]),color="green", alpha=0.8, label = "Curva teórica")
        #ax1.plot(xnew,power_smooth,color="red")
        ax1.plot(fitted_x, y,'g--', label='fit:a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(popt))
        ax1.set_xlim(0,1920)
        ax1.set_ylim(0,1080)
        ax1.set_ylabel("Pixeles eje y")
        ax1.set_xlabel("Pixeles eje x")
        ax1.legend(prop={'size': 22})
        ax1.grid()
        figManager = plt.get_current_fig_manager()
        figManager.set_window_title("Análisis de curvatura " + vid_name + " " + file_name[1:] + str(ds))
        figManager.window.showMaximized()
        plt.tight_layout()
        plt.savefig(file_path, dpi = 100)
        #plt.show()
        plt.close()

def graph_parameters(main_dir,vid_name,inclinations,vacuum,apparent_length,file_name="_Grafico_parametros"):
        file_path = main_dir + vid_name + file_name + ".png"
        plt.rc('axes', titlesize=12)     # fontsize of the axes title
        plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax1.set_title("Movimiento general del apéndice")
        ax1.plot(vacuum,inclinations,color="blue",label="Inclinación respecto a la horizontal")
        ax1.set_ylabel("Inclinación" + ' ['+ u'\N{DEGREE SIGN}' + ']')
        ax1.set_xlabel("Vacío [kPa]")
        ax1.legend(prop={'size': 7})
        ax1.grid()
        ax2.set_title("Largo aparente del apéndice")
        ax2.plot(vacuum,apparent_length,color="blue",label="Largo línea media")
        ax2.set_ylabel("Largo aparente [px]")
        ax2.set_xlabel("Vacío [kPa]")
        ax2.legend(prop={'size': 7})
        ax2.grid()
        figManager = plt.get_current_fig_manager()
        figManager.set_window_title(file_name + vid_name)
        figManager.window.showMaximized()
        plt.tight_layout()
        plt.savefig(file_path, dpi = 100)
        #plt.show()
        plt.close()

def graph_deflections(main_dir,vid_name,vacuum,list_deflect_de,list_deflect_ie,list_deflect_dt,list_deflect_it,list_deflect_c_dt,list_deflect_c_it,file_name="_deflexiones"):
    #Plot de los graficos
    file_path = main_dir + vid_name + file_name + ".png"
    fig, (ax1, ax2) = plt.subplots(2, 1) 
    ylim = max([max(list_deflect_de),max(list_deflect_ie),max(list_deflect_dt),max(list_deflect_it)])
    ax1.set_title("Deflexión aparente experimental")
    ax1.set_ylabel("Deflexión aparente [px]")
    ax1.set_xlabel("Vacío [kPa]")
    ax1.set_ylim(0, ylim)
    ax1.plot(vacuum, list_deflect_ie, color="orange", label="Deflexión izquierda")
    ax1.plot(vacuum, list_deflect_de, color="blue", label="Deflexión derecha")
    ax1.legend()
    ax2.set_title("Deflexión polyfit")
    ax2.set_ylabel("Deflexión [px]")
    ax2.set_xlabel("Vacío [kPa]")
    ax2.set_ylim(0, ylim)
    ax2.plot(vacuum, list_deflect_it, color="green", label="Deflexión izquierda")
    ax2.plot(vacuum, list_deflect_dt, color="magenta", label="Deflexión derecha")
    ax2.legend()
    #ax3.set_title("Deflexión calculada por parametros ")
    #ax3.set_xlabel("Deflexión [px]")
    #ax3.set_ylabel("Vacío[kPa]")
    #ax3.plot(list_deflect_c_it, vacuum,  color="c", label="deflexión izquierda")
    #ax3.plot(list_deflect_c_dt, vacuum,  color="y", label="deflexión derecha")
    #ax3.legend()
    #ax = plt.subplot(111)
    #ax.plot(diseño1_p, diseño1_fi, lw=2, color='orange')
    ax1.fill_between(vacuum, 0, list_deflect_ie, alpha=0.1, color='orange')
    ax1.fill_between(vacuum, 0, list_deflect_de, alpha=0.1, color='blue')
    ax2.fill_between(vacuum, 0, list_deflect_it, alpha=0.1, color='green')
    ax2.fill_between(vacuum, 0, list_deflect_dt, alpha=0.1, color='magenta')
    #ax3.fill_between(list_deflect_c_it, 0, vacuum, alpha=0.1, color='c')
    #ax3.fill_between(list_deflect_c_dt, 0, vacuum, alpha=0.1, color='y')
    ax1.grid()
    ax2.grid()
    #ax3.grid()
    figManager = plt.get_current_fig_manager()
    figManager.set_window_title(file_name + " " + vid_name)
    figManager.window.showMaximized()
    plt.tight_layout()
    plt.savefig(file_path, dpi = 100)
    #plt.show()
    plt.close()

def calc_vid_stats(list_RSE, list_deflect_de, list_deflect_dt, list_deflect_ie, list_deflect_it):
    residuals_d = []
    residuals_i = []
    video_RSE = np.mean(list_RSE)
    
    for i in range(len(list_deflect_de)):
        residuals_d.append(np.abs(list_deflect_de[i]-list_deflect_dt[i]))
        residuals_i.append(np.abs(list_deflect_ie[i]-list_deflect_it[i]))
    RSE_deflec_d = sum(residuals_d)/(len(list_deflect_de)-2)
    RSE_deflec_i = sum(residuals_i)/(len(list_deflect_ie)-2)
    
    return RSE_deflec_d, RSE_deflec_i, video_RSE


def save_report(RSE_deflec_d, RSE_deflec_i, video_RSE, main_dir, vid_name, file_name="_Estadisticas_dataset_"):
    #Create a dataframe to store the relevant information from the datasets
    file_loc = main_dir + vid_name + file_name + ".csv"
    print(file_loc,type(file_loc))
    data = {"RSE deflexion derecha": RSE_deflec_d,"RSE deflexion izquierda":RSE_deflec_i,"RSE Video":video_RSE}
    df = pd.DataFrame(data,index=[0])
    df.to_csv(file_loc)

def dataset_analisis(main_dir, vid_name, ds, file_name="\Dataset "):

    x, y = handle_dataset(main_dir, vid_name, ds, file_name)
    app_len = calc_distance((x[0],y[0]),(x[-1],y[-1]))
    fitted_x, popt, pcov = fit_to_poli(x,y)
    a,b,c,d = zip(popt)
    a = np.float64(a)[0]
    b = np.float64(b)[0]
    c = np.float64(c)[0]
    d = np.float64(d)[0]
    #Calculos de fitness
    RSS, RSE = calc_residual_stats(x, fitted_x)
    params_std = np.sqrt(np.diag(pcov))
    deflect_e, coord_emd, coord_emi, coord_ed, coord_ei,theta_inclination = analize_deflection_e(x,y)
    curv_radius_t, deflect_t, coord_td, coord_ti, coord_tmd, coord_tmi = analize_deflection_t(x,y,a,b,c,d)
    curv_radius_dt = curv_radius_t[0]
    curv_radius_it = curv_radius_t[1]
    deflect_de = deflect_e[0]
    deflect_ie = deflect_e[1]
    deflect_dt = deflect_t[0]
    deflect_it = deflect_t[1]
    #print(x,fitted_x,y,popt,coord_td, coord_ti, coord_tmd, coord_tmi, coord_emd, coord_emi, coord_ed, coord_ei)
    graph_dataset(main_dir, vid_name, ds, file_name,x,fitted_x,y,popt,coord_td, coord_ti, coord_tmd, coord_tmi, coord_emd, coord_emi, coord_ed, coord_ei)
    return RSS, RSE, params_std, deflect_de, deflect_dt, deflect_ie, deflect_it, curv_radius_dt, curv_radius_it,theta_inclination, app_len
        

def process_datasets(main_dir, vid_name,n_datasets = 25):
    list_RSS = []
    list_RSE = []
    list_params_std = []
    list_deflect_de = []
    list_deflect_dt = []
    list_deflect_ie = []
    list_deflect_it = []
    list_deflect_c_dt = []
    list_deflect_c_it = []
    list_theta_inclination = []
    list_app_len = []
    for ds in range(n_datasets):
        RSS, RSE, params_std, deflect_de, deflect_dt, deflect_ie, deflect_it, deflect_c_dt, deflect_c_it, theta_inclination, app_len = dataset_analisis(main_dir, vid_name,ds+1)
        list_RSS.append(RSS), list_RSE.append(RSE), list_params_std.append(params_std), list_deflect_de.append(deflect_de)
        list_deflect_dt.append(deflect_dt), list_deflect_ie.append(deflect_ie), list_deflect_it.append(deflect_it)
        list_deflect_c_dt.append(deflect_c_dt), list_deflect_c_it.append(deflect_c_it), list_theta_inclination.append(theta_inclination), list_app_len.append(app_len)
    
    RSE_deflec_d, RSE_deflec_i, video_RSE = calc_vid_stats(list_RSE, list_deflect_de, list_deflect_dt, list_deflect_ie, list_deflect_it)
    save_report(RSE_deflec_d, RSE_deflec_i, video_RSE, main_dir, vid_name)
    return list_deflect_de,list_deflect_ie,list_deflect_dt,list_deflect_it,list_deflect_c_dt,list_deflect_c_it, list_theta_inclination, list_app_len

def loop_videos(main_dir,vid_list,vacuum_csv, n_datasets = 25):
    vid_vacuums = pd.read_csv(vacuum_csv)
    for vid_name in vid_list:
        list_deflect_de,list_deflect_ie,list_deflect_dt,list_deflect_it,list_deflect_c_dt,list_deflect_c_it, list_theta_inclinations, list_app_len = process_datasets(main_dir, vid_name, n_datasets)
        vacuum = vid_vacuums[vid_name]
        graph_parameters(main_dir,vid_name,list_theta_inclinations,vacuum,list_app_len)
        graph_deflections(main_dir,vid_name,vacuum,list_deflect_de,list_deflect_ie,list_deflect_dt,list_deflect_it,list_deflect_c_dt,list_deflect_c_it)

    


#graph_dataset(folder,vid_folder,8)
######################################################### Function calls ##############################################################
std_datasets = []
loop_videos(main_folder,vid_names,vacuum_dir)