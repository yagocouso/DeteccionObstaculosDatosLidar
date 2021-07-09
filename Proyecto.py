import numpy as np 
import math
import random
import utm
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import vtk
from time import time

class vtkvisualizar:
    
    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e8):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.borrar_puntos()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)

    def añadir_punto(self, point):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def borrar_puntos(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')

def abrir_archivo(nombre_archivo):
        #Abre el archivo tipo bin
        NUBES= np.fromfile(nombre_archivo, np.float32).reshape(-1,4)
        return NUBES
    
def eliminar_intensidad(NUBE_COMPLETA):
    #Elimina la cuarta columna de la matriz de entrada
    FILTRO=np.array([True, True, True, False])
    NUBE_FINAL=NUBE_COMPLETA[:,FILTRO]
    return NUBE_FINAL
    
def guardar_archivo(ESCRITO,NOMBRE):
    #Guarda el archivo en documento de texto
    np.savetxt(NOMBRE,ESCRITO)

def mostrar_nube(valor):
    #Llama a la clase vtk para imprimir la nube
    puntonube = vtkvisualizar()
    filas_nube=valor.shape
    for x in range(filas_nube[0]):
        punto=valor[x,:]
        puntonube.añadir_punto(punto)
        
    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(puntonube.vtkActor)
    
    # renderer.SetBackground(.2, .3, .4)
    renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.ResetCamera()
 
    # Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
 
    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    
    # Comienza iteracion
    renderWindow.Render()
    renderWindowInteractor.Start()

def generar_nombre(i, EXTENSION):   
    #Genera nombres, necesita introducir la extensión
    NOMBRE=str(i)
    NOMBRE_FINAL=NOMBRE + EXTENSION
    while len(NOMBRE_FINAL)<10:
        NOMBRE_FINAL="0" + NOMBRE_FINAL
    return NOMBRE_FINAL

def abrir_trayectoria():
    #Carga la trayectoria
    TRAYECTORIA=np.loadtxt("0000.txt")
    return TRAYECTORIA

def limitar_trayectoria(TRAYECTORIA):
    #Limita la trayectoria para visulizarla
    VECTOR=[True,True,True]
    while len(VECTOR)<30:
        VECTOR.append(False)
    TRAYECTORIA_FINAL=TRAYECTORIA[:,VECTOR]
    return TRAYECTORIA_FINAL

def limitar_nube(NUBE_INICIO):
    #Reduce la nube a 25 metros
    NUBE=(NUBE_INICIO[:,0]**2 + NUBE_INICIO[:,1]**2)**0.5
    VECTOR=NUBE<25
    NUBE_REDUCIDA=NUBE_INICIO[VECTOR,:]
    return NUBE_REDUCIDA

def reducir_altura(NUBE_INICIO):
    #Reduce la altura de la nube
    APOYO=NUBE_INICIO[:,2]<0.77 
    NUBE_INICIO=NUBE_INICIO[APOYO,:]
    APOYO=NUBE_INICIO[:,2]>-2.63
    NUBE_REDUCIDA=NUBE_INICIO[APOYO,:]
    return NUBE_REDUCIDA

def segmentar_suelo(NUBE_COMPLETA):
    #Coge los puntos del suelo y devuelve el vector inverso
    APOYO_SUELO_2=NUBE_COMPLETA[:,2]<-1.43
    SUELO_SEGMENTADO=NUBE_COMPLETA[APOYO_SUELO_2,:]
    VECTOR_INVERSO=np.invert(APOYO_SUELO_2)
    return SUELO_SEGMENTADO, VECTOR_INVERSO

def puntos_vecinos(OBJETOS):
    #Calculo de puntos vecinos
    vecinos = NearestNeighbors(radius=1)
    vecinos.fit(OBJETOS)
    B,C=vecinos.radius_neighbors(OBJETOS,1,True)
    return C,B

def vecinos_centroides(MATRIZ_CENTROIDES, MINIMO):
    #Calcular cercania puntos centroides
    CERC=NearestNeighbors(n_neighbors=2)
    CERC.fit(MATRIZ_CENTROIDES)
    Vector=CERC.kneighbors(MATRIZ_CENTROIDES, return_distance=False)
    Vector=Vector[:,(False,True)]
    Vector.tolist
    Vector=Vector[:MINIMO]
    return Vector

def añadir_ceros(NUBE_SIN_CEROS,TAMAÑO):
    #Añade columna de ceros
    #NUBE_ALARGADA=np.pad(NUBE,((0, 0), (0, 1)), mode='constant', constant_values=0)
    NUBE_ALARGADA=np.column_stack([NUBE_SIN_CEROS,np.zeros(TAMAÑO)])
    return NUBE_ALARGADA

def añadir_unos(NUBE_SIN_UNOS):
    #Añade columna de unos
    NUBE_CON_UNOS=np.column_stack([NUBE_SIN_UNOS,np.ones((NUBE_SIN_UNOS.shape[0],1))])
    return NUBE_CON_UNOS

def agoritmo_agrupamiento(NUBE_NORMAL):
    #Une los objetos con el DBSCAN
    dbscan=DBSCAN(eps=0.5,min_samples=25).fit(NUBE_NORMAL)
    MEDIO=dbscan.labels_
    NUBE_OBJETOS=np.column_stack([NUBE_NORMAL,MEDIO])
    contador=max(MEDIO)
    return NUBE_OBJETOS,contador

def objeto_objeto(nube_modificada):
    #Genera la matriz de centroides
    Matriz=[]
    for i in range(int(max(nube_modificada[:,3]))):
        HOL=nube_modificada[:,3]==i
        fin=nube_modificada[HOL,:]
        fin=fin[:,(True,True,True,False)]
        Matriz=np.append(Matriz,clusterizado(fin))
    return Matriz

def clusterizado(NUBE):
    #Calcula los centroides de cada objeto
    CENTROIDE=np.mean(NUBE, axis=0)
    return CENTROIDE

def unir_centroides(MATRIZ_NUEVA,MATRIZ_VIEJA):
    #Combina los centroides
    CENTROIDES_UNIDOS=np.append(MATRIZ_NUEVA,MATRIZ_VIEJA)
    CENTROIDES_UNIDOS=np.array(CENTROIDES_UNIDOS).reshape(int(len(CENTROIDES_UNIDOS)/3),3)
    return CENTROIDES_UNIDOS

def rotar_traladar(NUBE,ANGULO,POSICION_TRASLACION):
    #Rota y traslada la nube
    #Matriz de rotación y traslacion
    MATRIZ_ROTAR_TRASLADAR=np.transpose(np.array([[math.cos(ANGULO), -math.sin(ANGULO),0,POSICION_TRASLACION[0]],[math.sin(ANGULO),math.cos(ANGULO),0,POSICION_TRASLACION[1]],[0,0,1,0],[0,0,0,1]]))
    NUBE_CON_UNOS=añadir_unos(NUBE)
    NUBE_ORIENTADA=NUBE_CON_UNOS @ MATRIZ_ROTAR_TRASLADAR
    NUBE_ORIENTADA=NUBE_ORIENTADA[:,(True,True,True,False)]
    return NUBE_ORIENTADA

def deter_movilidad(DIRECIONES_OBJETOS,DIRECION_TRAYECTORIA,LABELS):
    #Detecta si los vehiculos se mueven o no
    OBJETOS_FINALES=np.transpose(np.asarray([DIRECIONES_OBJETOS[:,0]-DIRECION_TRAYECTORIA[0],DIRECIONES_OBJETOS[:,1]-DIRECION_TRAYECTORIA[1]]))
    OBJETOS_FINALES=np.column_stack([OBJETOS_FINALES,LABELS])
    vector_estaticos=((DIRECIONES_OBJETOS[:,0]**2)+DIRECIONES_OBJETOS[:,1]**2)**(1/2)<0.35
    return vector_estaticos,OBJETOS_FINALES
  
def corte_trayectoria(MATRIZ_NUEVA_REDUCIDA,MATRIZ_TEMPORAL_REDUCIDA,DIRECION_TRAYECTORIA):
    #Calcula el punto de corte con la trayectoria
    UNOS_TEMPORAL=np.ones(MATRIZ_NUEVA_REDUCIDA.shape[0])
    X1,Y1=np.zeros(MATRIZ_NUEVA_REDUCIDA.shape[0]),np.zeros(MATRIZ_NUEVA_REDUCIDA.shape[0])
    X2,Y2=UNOS_TEMPORAL[:]*DIRECION_TRAYECTORIA[0],UNOS_TEMPORAL[:]*DIRECION_TRAYECTORIA[1]
    X3,Y3=MATRIZ_TEMPORAL_REDUCIDA[:,0],MATRIZ_TEMPORAL_REDUCIDA[:,1]
    X4,Y4=MATRIZ_NUEVA_REDUCIDA[:,0]+UNOS_TEMPORAL[:]*DIRECION_TRAYECTORIA[0],MATRIZ_NUEVA_REDUCIDA[:,1]+UNOS_TEMPORAL[:]*DIRECION_TRAYECTORIA[1]
    DESPLAZAMIENTO_OBJETO=((MATRIZ_NUEVA_REDUCIDA[:,0]-MATRIZ_TEMPORAL_REDUCIDA[:,0])**2+(MATRIZ_NUEVA_REDUCIDA[:,1]-MATRIZ_TEMPORAL_REDUCIDA[:,1])**2)**(0.5)
    DENOMINADOR=(X1-X2)*(Y3-Y4)-(Y1-Y2)*(X3-X4)
    PARALELOS=DENOMINADOR==0
    NO_PARALELOS=np.invert(PARALELOS)
    NUMERADOR_1=(X1*Y2-Y1*X2)*(X3-X4)-(X1-X2)*(X3*Y4-Y3*X4)
    NUMERADOR_2=(X1*Y2-Y1*X2)*(X3-X4)-(X1-X2)*(X3*Y4-Y3*X4)
    EJE_X=NUMERADOR_1[NO_PARALELOS]/DENOMINADOR[NO_PARALELOS]
    EJE_Y=NUMERADOR_2[NO_PARALELOS]/DENOMINADOR[NO_PARALELOS]
    FILTRO_1,FILTRO_2=EJE_Y>0,EJE_Y<25
    FILTRO=FILTRO_1*FILTRO_2
    DESPLAZAMIENTO_OBJETO=DESPLAZAMIENTO_OBJETO[NO_PARALELOS]
    DESPLAZAMIENTO_OBJETO=DESPLAZAMIENTO_OBJETO[FILTRO]
    #Incluye la velodidad en la reducion
    EJE_X_REDUCIDO,EJE_Y_REDUCIDO=EJE_X[FILTRO],EJE_Y[FILTRO]
    X2_RED,Y2_RED,X4_RED,Y4_RED=X2[FILTRO],Y2[FILTRO],X4[FILTRO],Y4[FILTRO]
    DISTANCIA_TRAYECT=((EJE_X_REDUCIDO-X2_RED)**2+(EJE_Y_REDUCIDO-Y2_RED)**2)**1/2
    DISTANCIA_OBJETO=((EJE_X_REDUCIDO-X4_RED)**2+(EJE_Y_REDUCIDO-Y4_RED)**2)**1/2
    VELOCIDAD_VEHICULO=(((DIRECION_TRAYECTORIA[0]**2+DIRECION_TRAYECTORIA[1]**2)**(1/2))*np.ones(DESPLAZAMIENTO_OBJETO.shape[0]))/((np.ones(DESPLAZAMIENTO_OBJETO.shape[0])*FRECUENCIA))
    VELOCIDAD_OBJETO=DESPLAZAMIENTO_OBJETO/(np.ones(DESPLAZAMIENTO_OBJETO.shape[0])*FRECUENCIA)
    DIFERENCIA=abs((DISTANCIA_TRAYECT/VELOCIDAD_VEHICULO)-(DISTANCIA_OBJETO/VELOCIDAD_OBJETO))    
    VECTOR_DIFERENCIA=DIFERENCIA<3
    return PARALELOS,FILTRO,VECTOR_DIFERENCIA

print("Acción a realizar (solo número):")
print("1 Mostrar nube completa")
print("2 Mostrar nube recortada")
print("3 Mostrar trayectoria")
print("4 Convertir a txt")
print("5 Mostrar suelo segmnetado")
print("6 Mostrar Objetos")
print("7 indices objetos")
print("8 Mostrar Matriz")
print("9 Comprobación de rotación nubes")
print("10 Imprimir direcion de trayectoria y matriz desplazamiento objetos")
print("11 Tiempos entre nubes")
print("12 Como son los objetos y si interfieren")

eleccion=int(input("Número: "))

if eleccion>8:
    COMIENZO_CUENTA=int(input("Nube de inicio: "))
    FIN_CUENTA=int(input("Nube de fin: "))
    PASO_CUENTA=int(input("Paso: "))

if eleccion == 1:
    mostrar_nube(eliminar_intensidad(abrir_archivo("000000.bin"))) 
    
elif eleccion ==2:
    mostrar_nube((reducir_altura(limitar_nube(eliminar_intensidad(abrir_archivo("000000.bin"))))))
    
elif eleccion==3:
    mostrar_nube(limitar_trayectoria(abrir_trayectoria()))
    
elif eleccion == 4:
    i=int(input("Numero de la nube (0-154): "))
    guardar_archivo(abrir_archivo(generar_nombre(i, ".bin")),generar_nombre(i,".txt"))
    
elif eleccion==5:
    NUBE_CORTADA=limitar_nube(reducir_altura(eliminar_intensidad(abrir_archivo("000000.bin"))))
    SUELO,VECTOR=segmentar_suelo(NUBE_CORTADA)
    mostrar_nube(SUELO)
    
elif eleccion==6:
    NUBE_CORTADA=limitar_nube(reducir_altura(eliminar_intensidad(abrir_archivo("000000.bin"))))
    SUELO,VECTOR=segmentar_suelo(NUBE_CORTADA)
    OBJETOS=NUBE_CORTADA[VECTOR,:]
    mostrar_nube(OBJETOS)
    
elif eleccion==7:
    start_time = time()
    NUBE_CORTADA=limitar_nube(reducir_altura(eliminar_intensidad(abrir_archivo("000000.bin"))))
    SUELO,VECTOR=segmentar_suelo(NUBE_CORTADA)
    OBJETOS=NUBE_CORTADA[VECTOR,:]
    NUBE=añadir_ceros(OBJETOS,OBJETOS.shape[0])
    FINAL,contador=agoritmo_agrupamiento(OBJETOS)
    objeto_objeto(FINAL)
    elapsed_time = time() - start_time
    print(FINAL)
    print("Elapsed time: %0.10f seconds." % elapsed_time)
    
elif eleccion==8:
    start_time = time()
    NUBE_CORTADA=limitar_nube(reducir_altura(eliminar_intensidad(abrir_archivo("000000.bin"))))
    SUELO,VECTOR=segmentar_suelo(NUBE_CORTADA)
    OBJETOS=NUBE_CORTADA[VECTOR,:]
    NUBE=añadir_ceros(OBJETOS,OBJETOS.shape[0])
    FINAL,contador=agoritmo_agrupamiento(OBJETOS)
    Matriz=np.array(objeto_objeto(FINAL)).reshape(contador,3)
    elapsed_time = time() - start_time
    print(Matriz)
    
elif eleccion==9:
    
    ANGULO=0
    DZ=0
    POSICION_TRAYECTORIA_VIEJA=[]
    DIRECION_TRAYECTORIA=[0,0,0]
    DESPLAZAMIENTOS_OBJETOS=[]
    MATRIZ_VIEJA=[]
    MATRIZ_NUEVA=[]
    TRAYECTORIA=abrir_trayectoria()
    DXY0= utm.from_latlon(TRAYECTORIA[0,0],TRAYECTORIA[0,1])
    DZ0=TRAYECTORIA[0,2]
    
    for i in range (COMIENZO_CUENTA,FIN_CUENTA,PASO_CUENTA):
    
        start_time = time()
        DXY= utm.from_latlon(TRAYECTORIA[i,0],TRAYECTORIA[i,1])
        DZ=TRAYECTORIA[i,2]
        POSICION_TRAYECTORIA_NUEVA=[DXY[0],DXY[1],DZ]
        POSICION_TRASLACION=[POSICION_TRAYECTORIA_NUEVA[0]-DXY0[0],POSICION_TRAYECTORIA_NUEVA[1]-DXY0[1],POSICION_TRAYECTORIA_NUEVA[2]-DZ0]
       
        if POSICION_TRAYECTORIA_VIEJA!=[]:
            DIRECION_TRAYECTORIA=[POSICION_TRAYECTORIA_NUEVA[0]-POSICION_TRAYECTORIA_VIEJA[0],POSICION_TRAYECTORIA_NUEVA[1]-POSICION_TRAYECTORIA_VIEJA[1],(POSICION_TRAYECTORIA_NUEVA[2]-POSICION_TRAYECTORIA_VIEJA[2])]
  
        NUBE_CORTADA=reducir_altura(limitar_nube(eliminar_intensidad(abrir_archivo(generar_nombre(i,".bin")))))
        NUBE_ORIENTADA=rotar_traladar(NUBE_CORTADA,TRAYECTORIA[i,5],POSICION_TRASLACION)
        SUELO,VECTOR=segmentar_suelo(NUBE_ORIENTADA)
        OBJETOS=NUBE_ORIENTADA[VECTOR,:]
        FINAL,contador=agoritmo_agrupamiento(OBJETOS)
        
        if i==0:
            guardar_archivo(NUBE_ORIENTADA,generar_nombre(i+1000,".txt"))
            
        if i==6:
            guardar_archivo(NUBE_ORIENTADA,generar_nombre(i+1000,".txt"))
        
        Matriz=np.array(objeto_objeto(FINAL)).reshape(contador,3)
        elapsed_time = time() - start_time
        Matriz=np.array(Matriz).reshape(contador,3)
        POSICION_TRAYECTORIA_VIEJA=POSICION_TRAYECTORIA_NUEVA
        print("Elapsed time: %0.10f seconds." % elapsed_time)
        #if i==2 or i==9:
          #guardar_archivo(convertir_nube(NUBE_TRASLADADA),generar_nombre(i,".txt"))
        
    elapsed_time = time() - start_time
    print("Elapsed time: %0.10f seconds." % elapsed_time)

elif eleccion==10:
    
    ANGULO=0
    DX=0
    DY=0
    DZ=0
    POSICION_TRAYECTORIA_VIEJA=[]
    DIRECION_TRAYECTORIA=[0,0,0]
    DESPLAZAMIENTOS_OBJETOS=[]
    MATRIZ_VIEJA=[]
    MATRIZ_NUEVA=[]
    DXY0= utm.from_latlon(TRAYECTORIA[0,0],TRAYECTORIA[0,1])
    DZ0=TRAYECTORIA[0,2]
    
    TRAYECTORIA=abrir_trayectoria()
    
    for i in range (COMIENZO_CUENTA,FIN_CUENTA,PASO_CUENTA):
    
        start_time = time()
        DXY= utm.from_latlon(TRAYECTORIA[i,0],TRAYECTORIA[i,1])
        DZ=TRAYECTORIA[i,2]
        POSICION_TRAYECTORIA_NUEVA=[DXY[0],DXY[1],DZ]
        POSICION_TRASLACION=[POSICION_TRAYECTORIA_NUEVA[0]-DXY0[0],POSICION_TRAYECTORIA_NUEVA[1]-DXY0[1],POSICION_TRAYECTORIA_NUEVA[2]-DZ0]
       
        if POSICION_TRAYECTORIA_VIEJA!=[]:
            DIRECION_TRAYECTORIA=[POSICION_TRAYECTORIA_NUEVA[0]-POSICION_TRAYECTORIA_VIEJA[0],POSICION_TRAYECTORIA_NUEVA[1]-POSICION_TRAYECTORIA_VIEJA[1],(POSICION_TRAYECTORIA_NUEVA[2]-POSICION_TRAYECTORIA_VIEJA[2])]
  
        NUBE_CORTADA=reducir_altura(limitar_nube(eliminar_intensidad(abrir_archivo(generar_nombre(i,".bin")))))
        NUBE_ORIENTADA=rotar_traladar(NUBE_CORTADA,TRAYECTORIA[i,5],POSICION_TRASLACION)
        SUELO,VECTOR=segmentar_suelo(NUBE_ORIENTADA)
        OBJETOS=NUBE_ORIENTADA[VECTOR,:]
        FINAL,contador=agoritmo_agrupamiento(OBJETOS)
        MATRIZ_NUEVA=np.array(objeto_objeto(FINAL)).reshape(contador,3)
        
        if POSICION_TRAYECTORIA_VIEJA!=[]:
            CENTROIDES_UNIDOS=unir_centroides(MATRIZ_NUEVA,MATRIZ_VIEJA)
            Vector=vecinos_centroides(CENTROIDES_UNIDOS,MATRIZ_NUEVA.shape[0])
            LABELS=np.arange(MATRIZ_NUEVA.shape[0])
            MATRIZ_TEMPORAL=CENTROIDES_UNIDOS[Vector,:].reshape(len(Vector),3)
            DIRECIONES_OBJETOS=np.column_stack([(MATRIZ_NUEVA[:,0]-MATRIZ_TEMPORAL[:,0]),(MATRIZ_NUEVA[:,1]-MATRIZ_TEMPORAL[:,1])])
        
        if POSICION_TRAYECTORIA_VIEJA!=[]:
            CENTROIDES_UNIDOS=np.append(MATRIZ_NUEVA,MATRIZ_VIEJA)
            CENTROIDES_UNIDOS=np.array(CENTROIDES_UNIDOS).reshape(int(len(CENTROIDES_UNIDOS)/3),3)
            Vector=vecinos_centroides(CENTROIDES_UNIDOS,MATRIZ_NUEVA.shape[0])
            MATRIZ_TEMPORAL=CENTROIDES_UNIDOS[Vector,:].reshape(len(Vector),3)
            DESPLAZAMIENTOS_OBJETOS=[MATRIZ_TEMPORAL[:,0]-MATRIZ_NUEVA[:,0],MATRIZ_TEMPORAL[:,1]-MATRIZ_NUEVA[:,1],MATRIZ_TEMPORAL[:,2]-MATRIZ_NUEVA[:,2]]
            DIRECIONES_OBJETOS=np.array(DESPLAZAMIENTOS_OBJETOS).reshape(len(Vector),3)
            
            OBJETOS_ESTATICOS=np.transpose(np.asarray([DIRECIONES_OBJETOS[:,0]-DIRECION_TRAYECTORIA[0],DIRECIONES_OBJETOS[:,1]-DIRECION_TRAYECTORIA[1]]))
            
            print(OBJETOS_ESTATICOS)
          
        MATRIZ_VIEJA=MATRIZ_NUEVA
        POSICION_TRAYECTORIA_VIEJA=POSICION_TRAYECTORIA_NUEVA

        elapsed_time = time() - start_time
        print("Elapsed time: %0.10f seconds." % elapsed_time)

    
elif eleccion==11:
    
    ANGULO=0
    DZ=0
    POSICION_TRAYECTORIA_VIEJA=[]
    DIRECION_TRAYECTORIA=[0,0,0]
    DESPLAZAMIENTOS_OBJETOS=[]
    MATRIZ_VIEJA=[]
    MATRIZ_NUEVA=[]
    TRAYECTORIA=abrir_trayectoria()
    DXY0= utm.from_latlon(TRAYECTORIA[0,0],TRAYECTORIA[0,1])
    DZ0=TRAYECTORIA[0,2]
    
    for i in range (COMIENZO_CUENTA,FIN_CUENTA,PASO_CUENTA):
    
        start_time = time()
        DXY= utm.from_latlon(TRAYECTORIA[i,0],TRAYECTORIA[i,1])
        DZ=TRAYECTORIA[i,2]
        POSICION_TRAYECTORIA_NUEVA=[DXY[0],DXY[1],DZ]
        POSICION_TRASLACION=[POSICION_TRAYECTORIA_NUEVA[0]-DXY0[0],POSICION_TRAYECTORIA_NUEVA[1]-DXY0[1],POSICION_TRAYECTORIA_NUEVA[2]-DZ0]
       
        if POSICION_TRAYECTORIA_VIEJA!=[]:
            DIRECION_TRAYECTORIA=[POSICION_TRAYECTORIA_NUEVA[0]-POSICION_TRAYECTORIA_VIEJA[0],POSICION_TRAYECTORIA_NUEVA[1]-POSICION_TRAYECTORIA_VIEJA[1],(POSICION_TRAYECTORIA_NUEVA[2]-POSICION_TRAYECTORIA_VIEJA[2])]
  
        NUBE_CORTADA=reducir_altura(limitar_nube(eliminar_intensidad(abrir_archivo(generar_nombre(i,".bin")))))
        NUBE_ORIENTADA=rotar_traladar(NUBE_CORTADA,TRAYECTORIA[i,5],POSICION_TRASLACION)
        SUELO,VECTOR=segmentar_suelo(NUBE_ORIENTADA)
        OBJETOS=NUBE_ORIENTADA[VECTOR,:]
        FINAL,contador=agoritmo_agrupamiento(OBJETOS)
        MATRIZ_NUEVA=np.array(objeto_objeto(FINAL)).reshape(contador,3)
        
        if POSICION_TRAYECTORIA_VIEJA!=[]:
            CENTROIDES_UNIDOS=unir_centroides(MATRIZ_NUEVA,MATRIZ_VIEJA)
            Vector=vecinos_centroides(CENTROIDES_UNIDOS,MATRIZ_NUEVA.shape[0])
            LABELS=np.arange(MATRIZ_NUEVA.shape[0])
            MATRIZ_TEMPORAL=CENTROIDES_UNIDOS[Vector,:].reshape(len(Vector),3)
            DIRECIONES_OBJETOS=np.column_stack([(MATRIZ_NUEVA[:,0]-MATRIZ_TEMPORAL[:,0]),(MATRIZ_NUEVA[:,1]-MATRIZ_TEMPORAL[:,1])])
            
            vector_estaticos,OBJETOS_FINALES=deter_movilidad(DIRECIONES_OBJETOS,DIRECION_TRAYECTORIA,LABELS)
            vector_moviles=np.invert(vector_estaticos)
            OBJETOS_ESTATICOS=OBJETOS_FINALES[vector_estaticos,:]          
            OBJETOS_MOVILES=OBJETOS_FINALES[vector_moviles,:]
            MATRIZ_NUEVA_REDUCIDA=MATRIZ_NUEVA[vector_moviles]
            MATRIZ_TEMPORAL_REDUCIDA=MATRIZ_TEMPORAL[vector_moviles]
            PARALELOS,FILTRO,VECTOR_DIFERENCIA=corte_trayectoria(MATRIZ_NUEVA_REDUCIDA,MATRIZ_TEMPORAL_REDUCIDA,DIRECION_TRAYECTORIA)
            OBJETOS_PARALELOS=OBJETOS_MOVILES[PARALELOS,:]
            OBJETOS_NO_PARALELOS=OBJETOS_MOVILES[np.invert(PARALELOS),:]
            OBJETOS_INTERVIENEN=OBJETOS_NO_PARALELOS[FILTRO,:]
            OBJETOS_COLISION=OBJETOS_INTERVIENEN[VECTOR_DIFERENCIA,:]
            
        MATRIZ_VIEJA=MATRIZ_NUEVA
        POSICION_TRAYECTORIA_VIEJA=POSICION_TRAYECTORIA_NUEVA

        elapsed_time = time() - start_time
        print("Elapsed time: %0.10f seconds." % elapsed_time)
        
elif eleccion==12:
    
    #TIEMPO ENTRE CADA NUBE
    FRECUENCIA=PASO_CUENTA*0.1
    ANGULO=0
    DZ=0
    POSICION_TRAYECTORIA_VIEJA=[]
    DIRECION_TRAYECTORIA=[0,0,0]
    DESPLAZAMIENTOS_OBJETOS=[]
    MATRIZ_VIEJA=[]
    MATRIZ_NUEVA=[]
    TRAYECTORIA=abrir_trayectoria()
    DXY0= utm.from_latlon(TRAYECTORIA[0,0],TRAYECTORIA[0,1])
    DZ0=TRAYECTORIA[0,2]
    
    for i in range (COMIENZO_CUENTA,FIN_CUENTA,PASO_CUENTA):
    
        start_time = time()
        DXY= utm.from_latlon(TRAYECTORIA[i,0],TRAYECTORIA[i,1])
        DZ=TRAYECTORIA[i,2]
        POSICION_TRAYECTORIA_NUEVA=[DXY[0],DXY[1],DZ]
        POSICION_TRASLACION=[POSICION_TRAYECTORIA_NUEVA[0]-DXY0[0],POSICION_TRAYECTORIA_NUEVA[1]-DXY0[1],POSICION_TRAYECTORIA_NUEVA[2]-DZ0]
       
        if POSICION_TRAYECTORIA_VIEJA!=[]:
            DIRECION_TRAYECTORIA=[POSICION_TRAYECTORIA_NUEVA[0]-POSICION_TRAYECTORIA_VIEJA[0],POSICION_TRAYECTORIA_NUEVA[1]-POSICION_TRAYECTORIA_VIEJA[1],(POSICION_TRAYECTORIA_NUEVA[2]-POSICION_TRAYECTORIA_VIEJA[2])]
  
        NUBE_CORTADA=reducir_altura(limitar_nube(eliminar_intensidad(abrir_archivo(generar_nombre(i,".bin")))))
        NUBE_ORIENTADA=rotar_traladar(NUBE_CORTADA,TRAYECTORIA[i,5],POSICION_TRASLACION)
        SUELO,VECTOR=segmentar_suelo(NUBE_ORIENTADA)
        OBJETOS=NUBE_ORIENTADA[VECTOR,:]
        FINAL,contador=agoritmo_agrupamiento(OBJETOS)
        MATRIZ_NUEVA=np.array(objeto_objeto(FINAL)).reshape(contador,3)
        
        if POSICION_TRAYECTORIA_VIEJA!=[]:
            CENTROIDES_UNIDOS=unir_centroides(MATRIZ_NUEVA,MATRIZ_VIEJA)
            Vector=vecinos_centroides(CENTROIDES_UNIDOS,MATRIZ_NUEVA.shape[0])
            LABELS=np.arange(MATRIZ_NUEVA.shape[0])
            MATRIZ_TEMPORAL=CENTROIDES_UNIDOS[Vector,:].reshape(len(Vector),3)
            DIRECIONES_OBJETOS=np.column_stack([(MATRIZ_NUEVA[:,0]-MATRIZ_TEMPORAL[:,0]),(MATRIZ_NUEVA[:,1]-MATRIZ_TEMPORAL[:,1])])
            
            vector_estaticos,OBJETOS_FINALES=deter_movilidad(DIRECIONES_OBJETOS,DIRECION_TRAYECTORIA,LABELS)
            print(vector_estaticos)
            vector_moviles=np.invert(vector_estaticos)
            
            OBJETOS_ESTATICOS=OBJETOS_FINALES[vector_estaticos,:]
            print("Los siguientes  objetos son estaticos:")
            print(OBJETOS_ESTATICOS[:,2])
            
            OBJETOS_MOVILES=OBJETOS_FINALES[vector_moviles,:]
            print("Los siguientes  objetos son moviles:")
            print(OBJETOS_MOVILES[:,2])
            
            MATRIZ_NUEVA_REDUCIDA=MATRIZ_NUEVA[vector_moviles]
            MATRIZ_TEMPORAL_REDUCIDA=MATRIZ_TEMPORAL[vector_moviles]
            PARALELOS,FILTRO,VECTOR_DIFERENCIA=corte_trayectoria(MATRIZ_NUEVA_REDUCIDA,MATRIZ_TEMPORAL_REDUCIDA,DIRECION_TRAYECTORIA,FRECUENCIA)
            
            OBJETOS_PARALELOS=OBJETOS_MOVILES[PARALELOS,:]
            OBJETOS_NO_PARALELOS=OBJETOS_MOVILES[np.invert(PARALELOS),:]
            
            print("Los siguientes objetos tienen una trayectoria paralela")
            print(OBJETOS_PARALELOS[:,2])
            
            print("Los siguientes objetos intervienen en la trayectoria")
            OBJETOS_INTERVIENEN=OBJETOS_NO_PARALELOS[FILTRO,:]
            print(OBJETOS_INTERVIENEN[:,2])
            
            print("Los siguentes obejtos tienen riesgo de colisión")
            OBJETOS_COLISION=OBJETOS_INTERVIENEN[VECTOR_DIFERENCIA,:]
            print(OBJETOS_COLISION[:,2])
            
        MATRIZ_VIEJA=MATRIZ_NUEVA
        POSICION_TRAYECTORIA_VIEJA=POSICION_TRAYECTORIA_NUEVA

        elapsed_time = time() - start_time
        print("Elapsed time: %0.10f seconds." % elapsed_time)

print("Terminado")
