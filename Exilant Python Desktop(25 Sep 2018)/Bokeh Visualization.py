##https://bokeh.pydata.org/en/latest/docs/gallery.html --> you click on the link and any of the graphs you get the source code of it

from bokeh.plotting import figure
from bokeh.io import  output_file,show

#prepare some data
x=[1,2,3,4,5]
y=[6,7,8,9,10]

#prepare the output file
output_file("Line.html")

#create a figure object
f=figure()

#create a line plot
f.line(x,y)
#display the plot in the figure object
show(f)
        
#Snippet producing the circle based plot       
f.circle(x,y)
show(f)

#Snippet producing the triangle based plot
f.triangle(x,y)
show(f) 

##BOKEH from CSV file
import pandas


df=pandas.read_csv("//Users//raghuram.b//Desktop//Python//Jupyter workings//data.csv")
x=df["x"]
y=df["y"]

output_file("bokeh_from_CSV.html")
f=figure()  
f.line(x,y)
show(f)

######################BOKEH plot with extra parameters#############
from bokeh.plotting import figure
from bokeh.io import output_file,show
import pandas
import xlrd ###for loading xls file


df=pandas.read_csv("//Users//raghuram.b//Desktop//Python//Jupyter workings//bachelors.xlsx")
x=df["Year"]
y=df["Engineering"]
output_file("bokeh_from_CSV.html")
f=figure(plot_width=1500,plot_height=400, tools='pan',logo=None)
f.title.text="Cool Data"
f.title.text_color="Gray"
f.title.text_font="times"
f.title.text_font_style="bold"
f.xaxis.minor_tick_line_color=None
f.yaxis.minor_tick_line_color=None
f.xaxis.axis_label="Date"
f.yaxis.axis_label="Engineering"    
#f.line([1,2,3],[4,5,6])    
f.line(x,y)
show(f)

#########BOKEH with EXCEL with extra parameters#######
from bokeh.plotting import figure
from bokeh.io import output_file,show
import pandas
import xlrd ###for loading xls file


df=pandas.read_excel("//Users//raghuram.b//Desktop//Python//Jupyter workings//verlegenhuken.xlsx",sheet_name=0)

x=df["Temperature"]/10
y=df["Pressure"]/10
output_file("Weather.html")
f=figure(plot_width=500,plot_height=400, tools='pan',logo=None)
f.title.text="Temperature and Air pressure"
f.title.text_color="Gray"
f.title.text_font="times"
f.title.text_font_style="bold"
f.xaxis.minor_tick_line_color=None
f.yaxis.minor_tick_line_color=None
f.xaxis.axis_label="Temperature(C)"
f.yaxis.axis_label="Pressure(hpa)"    
f.circle(x,y,size=0.5)
show(f)



####Bokeh with varying size of the circle points and include line graph too (2 graphs in one plot)
from bokeh.plotting import figure, output_file, show
p = figure(plot_width=500, plot_height=400, tools = 'pan, reset', logo=None)
p.title.text = "Earthquakes"
p.title.text_color = "Orange"
p.title.text_font = "times"
p.title.text_font_style = "italic"
p.yaxis.minor_tick_line_color = "Black"
p.xaxis.axis_label = "Times"
p.yaxis.axis_label = "Value"
p.line([20,7,8,9],[3,6,9,12])
p.circle([1,2,3,4,5], [5,6,5,5,3], size = [i*2 for i in [8,12,14,15,20]], color="red", alpha=0.5) ##Increasing the size of the circles
output_file("Scatter_plotting.html")
show(p)


###BOKEH plotting with date (as field in csv)
from bokeh.plotting import figure,output_file,show
import pandas

df=pandas.read_csv("//Users//raghuram.b//Desktop//Python//Jupyter workings//adbe.csv",parse_dates=["Date"])
 # If any date column in csv you need to mention it by parse_dates parameter

p=figure(width=500,height=250,x_axis_type="datetime")
#if its date you need to mention in x_axis_time as "datetime" else it gives output in exponential format

p.line(df["Date"],df["Close"],color="blue",alpha=0.7)

output_file("Timeseries.html")

show(p)


    
