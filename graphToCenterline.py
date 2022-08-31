import pickle
import vtk
import math
from collections import defaultdict
from collections import deque
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import dill

def write_polydata(input_data, filename, datatype=None):
    """
    Write the given input data based on the file name extension.
    Args:
        input_data (vtkSTL/vtkPolyData/vtkXMLStructured/
                    vtkXMLRectilinear/vtkXMLPolydata/vtkXMLUnstructured/
                    vtkXMLImage/Tecplot): Input data.
        filename (str): Save path location.
        datatype (str): Additional parameter for vtkIdList objects.
    """
    # Check filename format
    fileType = filename.split(".")[-1]
    if fileType == '':
        raise RuntimeError('The file does not have an extension')

    # Get writer
    if fileType == 'stl':
        writer = vtk.vtkSTLWriter()
    elif fileType == 'vtk':
        writer = vtk.vtkPolyDataWriter()
    elif fileType == 'vts':
        writer = vtk.vtkXMLStructuredGridWriter()
    elif fileType == 'vtr':
        writer = vtk.vtkXMLRectilinearGridWriter()
    elif fileType == 'vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    elif fileType == 'vtu':
        writer = vtk.vtkXMLUnstructuredGridWriter()
    elif fileType == "vti":
        writer = vtk.vtkXMLImageDataWriter()
    elif fileType == "np" and datatype == "vtkIdList":
        output_data = np.zeros(input_data.GetNumberOfIds())
        for i in range(input_data.GetNumberOfIds()):
            output_data[i] = input_data.GetId(i)
        output_data.dump(filename)
        return
    else:
        raise RuntimeError('Unknown file type %s' % fileType)

    # Set filename and input
    writer.SetFileName(filename)
    writer.SetInputData(input_data)
    writer.Update()

    # Write
    writer.Write()

class Graph:
	def __init__(self):
		self.nodes = set()
		self.edges = defaultdict(list)
		self.directed_edges = defaultdict(list)
		self.edge_nodes = defaultdict(set)
		self.edge_id = set()
		self.edge_length = dict()
		self.edge_radius = dict()
		self.lengths = dict()
		self.distances = {}
		self.outlets = set()
		self.radius = dict()
		self.nodes_xyz = dict()

	def add_node(self, value):
		self.nodes.add(value)

	def add_node_xyz(self, node, value):
		self.nodes_xyz[node] = value

	def add_outlet(self, value):
		self.outlets.add(value)

	def add_radius(self, node, value):
		self.radius[node] = value	

	def add_edge(self, from_node, to_node, distance):
		self.edges[from_node].append(to_node)
		self.edges[to_node].append(from_node)
		self.distances[(from_node, to_node)] = distance
		self.distances[(to_node, from_node)] = distance

	def add_edge_meta(self, from_node, to_node, length, radius, ID):
		self.edge_id.add(ID)
		self.edge_length[ID] = length
		self.edge_radius[ID] = radius
		self.edge_nodes[ID] = {from_node,to_node}
		self.directed_edges[from_node].append(to_node)
		self.distances[(from_node, to_node)] = length
		self.lengths[ID] = length

	def has_edge(self,from_node, to_node):
		if from_node in self.edges:
			if to_node in self.edges[from_node]:
				return True
			else:
				return False
		else:
			return False

	def has_node(self, node):
		if node in self.nodes:
			return True
		else:
			return False

	def add_virtual_node(self, v_node, zero_edge_nodes):
		self.nodes.add(v_node)
		for i in zero_edge_nodes:
			self.edges[v_node].append(i)
			self.edges[i].append(v_node)
			self.distances[(v_node, i)] = 0
			self.distances[(i, v_node)] = 0

	def add_virtual_node_distances(self, v_node, edge_nodes, distances):
		self.nodes.add(v_node)
		DISTANCE_CUTOFF = 1
		for i in edge_nodes:
			if(distances[i]<DISTANCE_CUTOFF):
				self.edges[v_node].append(i)
				self.edges[i].append(v_node)
				self.distances[(v_node, i)] = distances[i]
				self.distances[(i, v_node)] = distances[i]

	def get_num_of_nodes(self):
		return len(self.nodes)

	def get_nodes(self):
		return self.nodes

	def get_edge(self,edge):
		return self.edges[edge]

	def get_node_xyz(self, node):
		return self.nodes_xyz[node]

	def remove_outlet(self,outlet):
		if outlet not in self.outlets:
			print("ID not an outlet")
		self.outlets.remove(outlet)
		self.nodes.remove(outlet)
		self.nodes_xyz.pop(outlet)
		self.radius.pop(outlet)
		remove = set()
		for ID in self.edge_nodes:
			if outlet in self.edge_nodes[ID]:
				remove.add(ID)
		for ID in remove:
			self.edge_nodes.pop(ID)
			self.edge_length.pop(ID)
			self.edge_radius.pop(ID)
			self.edge_id.remove(ID)
		# connections = self.edges.pop(outlet)
		# for c in connections:
		# 	self.edges[c].remove(outlet)
		# 	self.outlets.add(c)



def calcDistance2Points(model, pt1,pt2):
	if(type(pt1) is int):
		x1,y1,z1 = model.GetPoint(pt1)
	elif(type(pt1) is list):
		x1,y1,z1 = pt1[0],pt1[1],pt1[2]
	else:
		print(type(pt1))
	if(type(pt2) is int):
		x2,y2,z2 = model.GetPoint(pt2)
	else:
		x2,y2,z2 = pt2[0],pt2[1],pt2[2]
	distance = ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**(.5)
	return distance

def getConnectedVerticesNotIncludingSeed(model, seedPt):
	cell_list = vtk.vtkIdList()
	connectedPts_list = vtk.vtkIdList()
	model.GetPointCells(seedPt,cell_list)
	for j in range(0,cell_list.GetNumberOfIds()):
		pt_list = vtk.vtkIdList()
		pt_list = model.GetCell(cell_list.GetId(j)).GetPointIds()
		for k in range(0,pt_list.GetNumberOfIds()):
			if (pt_list.GetId(k) != seedPt):
				connectedPts_list.InsertUniqueId(pt_list.GetId(k))
	return connectedPts_list

def generateGraph(heart):
	print('Generating graph...')
	heart_graph = Graph()
	edge_Id = 0
	print(heart.GetNumberOfPoints())
	for i in tqdm(range(0,heart.GetNumberOfPoints())):
		connnectedPt_list = getConnectedVerticesNotIncludingSeed(heart,i)
		if connnectedPt_list.GetNumberOfIds()>0:
			heart_graph.add_node(i)
			heart_graph.add_node_xyz(i,heart.GetPoint(i))
			if connnectedPt_list.GetNumberOfIds()==1:
				heart_graph.add_outlet(i)
			for j in range(0,connnectedPt_list.GetNumberOfIds()):
				cpt = connnectedPt_list.GetId(j)
				if not heart_graph.has_edge(i,cpt):
					# new point to decide whether to add to patch, edge, or nothing (if already in edge)
					heart_graph.add_radius(i,heart.GetPointData().GetArray('Radius').GetValue(i))
					heart_graph.add_radius(cpt,heart.GetPointData().GetArray('Radius').GetValue(cpt))
					heart_graph.add_edge(i,cpt,calcDistance2Points(heart,i,cpt))
	return heart_graph

def dijkstra(graph, initial):
	visited = {}
	visited[initial] = 0
	path = {}
	path_nodes = set()
	nodes = set(graph.nodes)
	counter = 0
	pbar = tqdm(total=len(nodes))
	while nodes: 
		pbar.update(1)
		min_node = None
		for node in nodes:
			if node in visited:
				if min_node is None:
					min_node = node
				elif visited[node] <= visited[min_node]:
					min_node = node
		if min_node is None:
			break

		nodes.remove(min_node)
		current_weight = visited[min_node]

		for edge in graph.edges[min_node]:
			weight = current_weight + graph.distances[(min_node, edge)]
			if edge not in visited or weight <= visited[edge]:
				visited[edge] = weight
				path[edge] = min_node
				path_nodes.add(edge)
		counter += 1
	pbar.close()
	return visited, path

def dijkstra_label(graph, initial):
	visited = {}
	visited[initial] = 0
	path = {}
	path_nodes = set()
	nodes = set(graph.nodes)
	counter = 0
	pbar = tqdm(total=len(nodes))
	while nodes: 
		pbar.update(1)
		min_node = None
		for node in nodes:
			if node in visited:
				if min_node is None:
					min_node = node
				elif visited[node] <= visited[min_node]:
					min_node = node
		if min_node is None:
			print('break')
			break

		nodes.remove(min_node)
		current_weight = visited[min_node]
		for edge in graph.edges[min_node]:
			weight = current_weight + graph.distances[(min_node, edge)]
			if edge not in visited or weight <= visited[edge]:
				graph.add_edge_meta(min_node,edge,graph.distances[(min_node, edge)], (graph.radius[min_node]+graph.radius[edge])/2, counter)
				visited[edge] = weight
				path[edge] = min_node
				path_nodes.add(edge)
				counter += 1
	pbar.close()
	return visited, path



def dijsktra_closest(graph, initial, destinations):
	visited = {initial: 0}
	path = {}
	path_nodes = set()
	nodes = set(graph.nodes)
	parent = [initial]
	if initial in destinations:
		destinations.remove(initial)

	while len(destinations.intersection(path_nodes))==0: 
		min_node = None
		for node in nodes:
			if node in visited:
				if min_node is None:
					min_node = node
				elif visited[node] < visited[min_node]:
					min_node = node
		#print(min_node)
		if min_node is None:
			break

		nodes.remove(min_node)
		current_weight = visited[min_node]

		for edge in graph.edges[min_node]:
			weight = current_weight + graph.distances[(min_node, edge)]
			if edge not in visited or weight < visited[edge]:
				visited[edge] = weight
				path[edge] = min_node
				path_nodes.add(min_node)

	return visited, path, destinations.intersection(path)

def weightedDijsktra(graph, initial, weights):
	visited = {}
	visited[initial] = 0
	path = {}
	path_nodes = set()

	nodes = set(graph.nodes)
	counter = 0
	vprint('Starting to build distance map...')
	pbar = tqdm(total=len(nodes))
	while nodes: 
		pbar.update(1)
		min_node = None
		for node in nodes:
			if node in visited:
				if min_node is None:
					min_node = node
				elif visited[node] <= visited[min_node]:
					min_node = node
		if min_node is None:
			break

		nodes.remove(min_node)
		current_dist = visited[min_node]

		for edge in graph.edges[min_node]:
			if min_node==initial:
				dist = current_dist + graph.distances[(min_node, edge)]
			else:
				dist = current_dist + graph.distances[(min_node, edge)]*weights[min_node]
			if edge not in visited or dist <= visited[edge]:
				visited[edge] = dist
				path[edge] = min_node
				path_nodes.add(edge)
				if min_node in weights:
					weights[edge] = weights[min_node]
		counter += 1
	pbar.close()
	return visited, path	

def shortest_path(visited, paths, origin, destination):
	full_path = deque()
	_destination = paths[destination]

	while _destination != origin:
		print(_destination)
		full_path.appendleft(_destination)
		_destination = paths[_destination]

	full_path.appendleft(origin)
	full_path.append(destination)

	return visited[destination], list(full_path)

def shortest_path_(graph, path, origin, destination):
	global short


	while _destination != origin:
		full_path.appendleft(_destination)
		_destination = paths[_destination]

	full_path.appendleft(origin)
	full_path.append(destination)

	return visited[destination], list(full_path)

def makeVTK(version,graph=None,radii=None,conn=None,xyz=None):
	
	g = vtk.vtkMutableUndirectedGraph()
	if graph is None:
		points = vtk.vtkPoints()
		for i in xyz:
			points.InsertNextPoint(i)
			g.AddVertex()

		for edge in conn:
			g.AddEdge(edge[0],edge[1])

		g.SetPoints(points)
	else:
		points = vtk.vtkPoints()
		for i in sorted(list(graph.nodes)):
			points.InsertNextPoint(graph.nodes_xyz[i])
			g.AddVertex()

		g.SetPoints(points)
		print(len(list(graph.edge_id)))
		for edge in graph.edge_id:
			edges = list(graph.edge_nodes[edge])
			g.AddEdge(edges[0],edges[1])

	graphToPolyData = vtk.vtkGraphToPolyData()
	graphToPolyData.SetInputData(g)
	graphToPolyData.Update()
	cl = graphToPolyData.GetOutput()

	vtk_id = vtk.vtkDoubleArray()
	for i in range(0,cl.GetNumberOfPoints()):
		vtk_id.InsertNextValue(i)
	vtk_id.SetName('Id')
	cl.GetPointData().AddArray(vtk_id)

	if graph is None:
		if radii is not None:
			vtk_radii = vtk.vtkDoubleArray()
		for i in radii:
			vtk_radii.InsertNextValue(i)
		vtk_radii.SetName('Radius')
		cl.GetPointData().AddArray(vtk_radii)
	else:
		vtk_edge_radii = vtk.vtkDoubleArray()
		print(len(list(graph.edge_radius)))
		for i in sorted(list(graph.edge_radius.keys())):
			vtk_edge_radii.InsertNextValue(graph.edge_radius[i])
		vtk_edge_radii.SetName('Edge_Radius')
		cl.GetCellData().AddArray(vtk_edge_radii)

		vtk_edge_id = vtk.vtkDoubleArray()
		print(len(list(graph.edge_id)))
		for i in sorted(list(graph.edge_id)):
			vtk_edge_id.InsertNextValue(i)
		vtk_edge_id.SetName('Edge_ID')
		cl.GetCellData().AddArray(vtk_edge_id)

		vtk_radii = vtk.vtkDoubleArray()
		for i in sorted(list(graph.radius.keys())):
			vtk_radii.InsertNextValue(graph.radius[i])
		vtk_radii.SetName('Radius')
		cl.GetPointData().AddArray(vtk_radii)

		cleanPolyData = vtk.vtkCleanPolyData()
		cleanPolyData.SetInputData(cl)
		cleanPolyData.Update()
		cl = cleanPolyData.GetOutput()

	writer = vtk.vtkXMLPolyDataWriter()
	writer.SetInputData(graphToPolyData.GetOutput())
	writer.SetFileName('graph_'+version+'.vtp')
	writer.Write()

	return g,cl

def writeVesselTreeMetrics(graph):
	junctions = dict()
	vessel_outlets = set()
	for i in sorted(list(graph.nodes)):
		junction_edges = set()
		
		if i not in graph.outlets:
			print(i)
			for e in graph.edge_id:
				if i in graph.edge_nodes[e]:
					print(graph.edge_nodes[e])
					#for n in graph.edge_nodes[e]:
					junction_edges.add(e)
			parent = min(junction_edges)
			junction_edges.remove(parent)
			children = list(junction_edges)
			junctions[parent] = children
		else:
			print(str(i)+' is an outlet.')
			for e in graph.edge_nodes:
				if i in graph.edge_nodes[e]:
					vessel_outlets.add(e)
	print(junctions)
	pickle.dump(junctions,open('junctions.p','wb'))
	pickle.dump(graph.edge_id,open('vessels.p','wb'))
	pickle.dump(vessel_outlets,open('outlets.p','wb'))
	pickle.dump(graph.edge_radius,open('radius.p','wb'))
	pickle.dump(graph.edge_length,open('length.p','wb'))

def writeVesselTreeMetrics_cl(cl):
	junctions = dict()
	vessels = set()
	outlets = set()
	radius = dict()
	length = dict()

	NumCells = cl.GetNumberOfCells()
	NumPts = cl.GetNumberOfPoints()

	for p in range(0,NumPts):
		cell_list = vtk.vtkIdList()
		connectedPts_list = vtk.vtkIdList()
		cl.GetPointCells(p,cell_list)
		print(getConnectedVerticesNotIncludingSeed(cl, p).GetNumberOfIds())
		if cell_list.GetNumberOfIds()>1:
			edge = set()
			for j in range(0,cell_list.GetNumberOfIds()):
				vessels.add(cell_list.GetId(j))
				edge.add(cell_list.GetId(j))
				pt_list = vtk.vtkIdList()
				pt_list = cl.GetCell(cell_list.GetId(j)).GetPointIds()
				if pt_list.GetNumberOfIds()==2:
					p1 = pt_list.GetId(0)
					p2 = pt_list.GetId(1)
					#get radius
					r1 = cl.GetPointData().GetArray('Radius').GetValue(p1)
					r2 = cl.GetPointData().GetArray('Radius').GetValue(p2)
					radius[cell_list.GetId(j)] = (r1+r2)/2
					#get length
					length[cell_list.GetId(j)] = calcDistance2Points(cl,int(p1),int(p2))
				else:
					print('error')
					exit()

			parent = min(edge)
			print(p,edge)
			edge.remove(parent)
			children = edge
			junctions[parent] = sorted(list(children))
		elif cell_list.GetNumberOfIds()==1:
			outlets.add(cell_list.GetId(0))
			vessels.add(cell_list.GetId(0))

	correctedJunctions = dict()
	for j in range(0,len(list(junctions.keys()))):
		correctedJunctions[j] = {'inlet':[list(junctions.keys())[j]],'outlet':junctions[list(junctions.keys())[j]]}

	#print(correctedJunctions[4225])

	out = set()
	collaterals = set()
	values = list(junctions.values())
	keys = list(junctions.keys())
	for i in range(0,len(values)):
		for j in values[i]:
			if j in out:
				#print('duplicate',j)
				#print(out,j)
				collaterals.add(j)
			out.add(j)

	collaterals = set()
	out_dict = dict()
	in_dict = dict()
	col_nodes = []
	for i in correctedJunctions:
		for j in correctedJunctions[i]['outlet']:
			if j in list(out_dict.keys()):
				collaterals.add(j)
				print('duplicate out',i,j,out_dict[j])
				col_nodes.append([i,out_dict[j]])
			out_dict[j] = i
		for j in correctedJunctions[i]['inlet']:
			if j in list(in_dict.keys()):
				collaterals.add(j)
				print('duplicate in',i,j,in_dict[j])
			in_dict[j] = i
	print(collaterals)
	print(len(collaterals))
	for d in list(collaterals):
		print(d)
		collateral_junctions = []
		for j in correctedJunctions:
			for o in correctedJunctions[j]['outlet']:
				if o==d:
					collateral_junctions.append(j)
		sorted(collateral_junctions)
		print(collateral_junctions)
		if len(list(correctedJunctions[collateral_junctions[0]]['outlet']))>1 and len(list(correctedJunctions[collateral_junctions[0]]['inlet']))>0:
			print('before',correctedJunctions[collateral_junctions[0]]['inlet'])
			curr_inlet = correctedJunctions[collateral_junctions[0]]['inlet'][0]
			correctedJunctions[collateral_junctions[0]]['inlet'] = [curr_inlet,d]
			print('after',correctedJunctions[collateral_junctions[0]]['inlet'])
			outs = set(correctedJunctions[collateral_junctions[0]]['outlet'])
			outs.remove(d)
			correctedJunctions[collateral_junctions[0]]['outlet'] = list(outs)
		elif len(list(correctedJunctions[collateral_junctions[1]]['outlet']))>1 and len(list(correctedJunctions[collateral_junctions[1]]['inlet']))>0:
			print('before',correctedJunctions[collateral_junctions[1]]['inlet'])
			curr_inlet = correctedJunctions[collateral_junctions[1]]['inlet'][0]
			correctedJunctions[collateral_junctions[1]]['inlet'] = [curr_inlet,d]
			print('after',correctedJunctions[collateral_junctions[1]]['inlet'])
			outs = set(correctedJunctions[collateral_junctions[1]]['outlet'])
			outs.remove(d)
			correctedJunctions[collateral_junctions[1]]['outlet'] = list(outs)
		else:
			print('only 1 for ', d)
			print(list(correctedJunctions[collateral_junctions[0]]['inlet']))
			print(list(correctedJunctions[collateral_junctions[1]]['inlet']))
			print(list(correctedJunctions[collateral_junctions[0]]['outlet']))
			print(list(correctedJunctions[collateral_junctions[1]]['outlet']))
			#set curr_juction inlet
			curr_junction = [collateral_junctions[0]]
			while len(correctedJunctions[curr_junction[-1]]['outlet'])==1:
				curr_inlet = correctedJunctions[curr_junction[-1]]['inlet'][0]
				curr_outlet = correctedJunctions[curr_junction[-1]]['outlet']
				
				#search for junction that has an outlet of the curr_juction's inlet
				for i in correctedJunctions:
					if curr_inlet in correctedJunctions[i]['outlet']:
						curr_junction.append(i)
				if correctedJunctions[curr_junction[-1]]['inlet'][0]==curr_inlet:
					print('Error')
					break
				#set curr_juction
				print(curr_junction)
				print(correctedJunctions[curr_junction[-1]])		
			correctedJunctions[curr_junction[-1]]['inlet'].append(curr_inlet)
			outs = set(correctedJunctions[curr_junction[-1]]['outlet'])
			outs.remove(curr_inlet)
			correctedJunctions[curr_junction[-1]]['outlet'] = list(outs)
			for i in range(0,len(curr_junction)-1):
				curr_in = correctedJunctions[curr_junction[i]]['inlet']
				curr_out = correctedJunctions[curr_junction[i]]['outlet']
				correctedJunctions[curr_junction[i]]['inlet'] = curr_out
				correctedJunctions[curr_junction[i]]['outlet'] = curr_in
				print(curr_junction)
				print(correctedJunctions[curr_junction[i]])
			print(curr_junction)
			print(correctedJunctions[curr_junction[-1]])

	total_out = 0
	total_in = 0
	out_dict = dict()
	in_dict = dict()
	for i in correctedJunctions:
		total_out += len(list(correctedJunctions[i]['outlet']))
		total_in += len(list(correctedJunctions[i]['inlet']))
		for j in correctedJunctions[i]['outlet']:
			if j in list(out_dict.keys()):
				print('duplicate out',i,j,out_dict[j])
			out_dict[j] = i
		for j in correctedJunctions[i]['inlet']:
			if j in list(in_dict.keys()):
				print('duplicate in',i,j,in_dict[j])
			in_dict[j] = i
		if len(correctedJunctions[i]['outlet'])==0:
			print('empty_outlet',i)

	print('outlets',total_out)
	print('inlets',total_in)

	total = 0
	for i in junctions:
		total += len(list(junctions[i]))
	#print(correctedJunctions)
	print(len(junctions.keys()))
	print(len(outlets))
	for v in vessels:
		in_junc = []
		out_junc = []
		for j in correctedJunctions:
			if v in correctedJunctions[j]['inlet']:
				in_junc.append(j)
			if v in correctedJunctions[j]['outlet']:
				out_junc.append(j)
		if len(in_junc)>1:
			print('vessel ',v,'junction',in_junc,'not single')
		if len(out_junc)>1:
			print('vessel ',v,'junction',out_junc,'not single')


	pickle.dump(junctions,open('junctions.p','wb'))
	pickle.dump(correctedJunctions,open('correctedJunctions.p','wb'))
	pickle.dump(vessels,open('vessels.p','wb'))
	pickle.dump(outlets,open('outlets.p','wb'))
	pickle.dump(radius,open('radius.p','wb'))
	pickle.dump(length,open('length.p','wb'))
	pickle.dump(collaterals,open('collaterals.p','wb'))
	pickle.dump(col_nodes,open('col_nodes.p','wb'))
	return junctions,correctedJunctions,vessels,outlets,radius,length, collaterals

def erodeTerminalBranches(graph,size=2,mode=None):
	removed = set()
	if mode=='larger':
		for o in graph.outlets:
			if graph.radius[o] > size:
				removed.add(o)
	if mode=='smaller':
		for o in graph.outlets:
			if graph.radius[o] < size:
				removed.add(o)
	if mode=='equal':
		for o in graph.outlets:
			if graph.radius[o] == size:
				removed.add(o)
	for remove in removed:
		graph.remove_outlet(remove)
	return removed

def erodeTerminalBranches_cl(cl,size=2,mode=None):
	outlets = set()
	remove = set()
	for pt in range(cl.GetNumberOfPoints()):
		connnectedPt_list = getConnectedVerticesNotIncludingSeed(cl,pt)
		if connnectedPt_list.GetNumberOfIds()==1:
			outlets.add(pt)
	if mode=='larger':
		for o in outlets:
			radius = cl.GetPointData().GetArray('Radius').GetValue(o)
			if radius > size:
				remove.add(o)
				connnectedPt_list = getConnectedVerticesNotIncludingSeed(cl,pt)
				while connnectedPt_list.GetNumberOfIds()>1:
					temp_list = []
					for cpt in range(0,connnectedPt_list.GetNumberOfIds()):
						cpt_id = connnectedPt_list.GetId(cpt)
						radius = cl.GetPointData().GetArray('Radius').GetValue(cpt_id)
						if radius > size and cpt_id:
							remove.add(cpt_id)
							temp_list.append(cpt_id)
					connnectedPt_list = vtk.vtkIdList()
					for i in temp_list:
						connnectedPt_list.InsertUniqueId(i)

	if mode=='smaller':
		for o in graph.outlets:
			if graph.radius[o] < size:
				removed.add(o)
	if mode=='equal':
		for o in graph.outlets:
			if graph.radius[o] == size:
				removed.add(o)
	dataArray = vtk.vtkDoubleArray()
	for pt in range(cl.GetNumberOfPoints()):
		if pt in remove:
			dataArray.InsertNextValue(1)
		else:
			dataArray.InsertNextValue(0)
	dataArray.SetName('Eroded')
	cl.GetPointData().AddArray(dataArray)

# (C) Classify each segment into an initial order using the Strahler Ordering System
#	Terminal branches = order 1
#	When two branches of order n converge to 1 branch, converged branch = order n+1
#	If a higher order branch converges with a lower order branch, converged branch = higher order
def initStrahler(segDiam, segOrder, maxOrder, inletSegName, cl):
	# Define terminal branches as the end segment with no more joints (attached segment) as order 1
	# print("\n\nBefore Strahler Ordering")
	print(inletSegName)
	for seg in segName:
		if cl:
			if len(jointSeg[seg])<1:
				segOrder[seg] = 1
			# print(str(seg) + " " + str(segName[seg][0]) + " " + str(segOrder[int(segName[seg][0])]) )
		else:
			segOrder[seg] = 1
			# print(str(seg) + " " + str(segName[seg][-1]) + " " + str(segOrder[int(segName[seg][-1])]) )
			

	parentSeg = segName[inletSegName] #min(set(jointSeg)) # Str of Model's inlet segment
	maxGen = 0
	[segOrder, maxGen] = findStrahOrders(segOrder, parentSeg, maxGen)


	print("MAX GENERATION = " + str(maxGen))

	# print("\n\nAfter Strahler Ordering")
	for seg in segName:
		if cl:
			if not jointSeg[seg]:
				segOrder[seg] = 1
			# print(str(seg) + " " + str(segName[seg][0]) + " " + str(segOrder[int(segName[seg][0])]) )
		else:
			segOrder[seg] = 1
			# print(str(seg) + " " + str(segName[seg][-1]) + " " + str(segOrder[int(segName[seg][-1])]) )
			

	# print("\n\nJOINT SEG")
	# for vessel in segName:
	# 	for seg in segName[vessel]:
	# 		print(str(vessel) + " " + str(seg) + ": " + str(jointSeg[seg]))

	maxinitOrder = max(segOrder.values())
	orderTransl = maxOrder-maxinitOrder # Assumes MPA included
	for seg in segOrder:
		segOrder[seg] = segOrder[seg] + orderTransl


	return(segOrder, maxGen)

# Recursively find the orders of all segments in model
def findStrahOrders(segOrder, parentSeg, maxGen):
	d_orders = []

	# Track max generation number in model: each bifurcation adds 1 generation definition
	gen_num = []

	# Find Daughter Orders
	#print(jointSeg[parentSeg])
	for daughterSeg in jointSeg[parentSeg]: # find daughter segments of parent seg
		# If daughter not defined, make it the new parent to find orders below
		if segOrder[daughterSeg] == 0: 
			ordersNotDefined = True
			
			[segOrder, maxGen] = findStrahOrders(segOrder, daughterSeg, maxGen) # recursively call down the tree
			gen_num.append(maxGen)

			# Once all daughter orders found, backs out and can now define current daughter's order and append
			d_orders.append(segOrder[daughterSeg])
		else: # Append daughter's order to list of daughter orders
			d_orders.append(segOrder[daughterSeg])
			maxGen = 0
			gen_num.append(maxGen)
	
	# Determine the Parent Order based on daughter order
	#print('segOrder=',len(segOrder))
	#print('parentSeg=',parentSeg)
	#print('d_orders=',len(d_orders))
	if len(set(d_orders)) == 1 and len(jointSeg[parentSeg]) > 1: # if all daughters' orders the same, increase parent order by 1
		segOrder[int(parentSeg)] = d_orders[0]+1
	elif len(set(d_orders)) < 1:
		segOrder[int(parentSeg)] = 1
	else: # if daughter orders not the same, parent order is the largest daughter order
		segOrder[int(parentSeg)] = max(d_orders)

	# Only add generation at end of parent bifurcation for the largest gen #
	#maxGen = max(gen_num)
	maxGen += 1
	

	return(segOrder, maxGen)

# Read a vtp file and return the polydata
def read_polydata(filename, datatype=None):
    """
    Load the given file, and return a vtkPolyData object for it.
    Args:
        filename (str): Path to input file.
        datatype (str): Additional parameter for vtkIdList objects.
    Returns:
        polyData (vtkSTL/vtkPolyData/vtkXMLStructured/
                    vtkXMLRectilinear/vtkXMLPolydata/vtkXMLUnstructured/
                    vtkXMLImage/Tecplot): Output data.
    """

    # Check if file exists
    if not os.path.exists(filename):
        raise RuntimeError("Could not find file: %s" % filename)

    # Check filename format
    fileType = filename.split(".")[-1]
    if fileType == '':
        raise RuntimeError('The file does not have an extension')

    # Get reader
    if fileType == 'stl':
        reader = vtk.vtkSTLReader()
        reader.MergingOn()
    elif fileType == 'vtk':
        reader = vtk.vtkPolyDataReader()
    elif fileType == 'vtp':
        reader = vtk.vtkXMLPolyDataReader()
    elif fileType == 'vts':
        reader = vtk.vtkXMinkorporereLStructuredGridReader()
    elif fileType == 'vtr':
        reader = vtk.vtkXMLRectilinearGridReader()
    elif fileType == 'vtu':
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif fileType == "vti":
        reader = vtk.vtkXMLImageDataReader()
    elif fileType == "np" and datatype == "vtkIdList":
        result = np.load(filename).astype(np.int)
        id_list = vtk.vtkIdList()
        id_list.SetNumberOfIds(result.shape[0])
        for i in range(result.shape[0]):
            id_list.SetId(i, result[i])
        return id_list
    else:
        raise RuntimeError('Unknown file type %s' % fileType)

    # Read
    reader.SetFileName(filename)
    reader.Update()
    polydata = reader.GetOutput()

    return polydata

def label_outlier_vessels(cl,segOrder,segDiam,length):
	OrderDiam = defaultdict(list)
	OrderSeg = defaultdict(list)
	DiamSeg = dict()
	for i in segOrder:
		OrderDiam[segOrder[i]].append(segDiam[i])
		OrderSeg[segOrder[i]].append(i)
		DiamSeg[segDiam[i]] = i
	avgDiam = dict()
	stdDiam = dict()
	numOrder = dict()
	for order in OrderSeg:
		avgDiam[order] = np.mean(np.asarray(OrderDiam[order]))
		stdDiam[order] = np.std(np.asarray(OrderDiam[order]))
		numOrder[order] = len(np.asarray(OrderDiam[order]))
	print('avgDiam',avgDiam)
	print('stdDiam',stdDiam)
	print('numOrder',numOrder)
	to_remove = set()
	for seg in segOrder:
		Diam = segDiam[seg]
		order = segOrder[seg]
		if (order < 8 and Diam > 20) or (order < 3 and length[seg] < 30) or (order < 7 and Diam > 15):
			to_remove.add(seg)
	print(len(list(to_remove)))
	data = vtk.vtkDoubleArray()
	for cell in range(cl.GetNumberOfCells()):
		if cell in to_remove:
			data.InsertNextValue(1)
		else:
			data.InsertNextValue(0)
	data.SetName('to_remove')
	cl.GetCellData().AddArray(data)


def main():
	file = 'graph_iter3.vtu'

	genGraph = True
	if file is None:
		version = 'reduced'
		radii = pickle.load(open('HU19_threshold0.45_min5000_max67000_fullimageradii_'+version+'.p','rb'))
		conn = pickle.load(open('HU19_threshold0.45_min5000_max67000_fullimageconnectivity_'+version+'.p','rb'))
		xyz = pickle.load(open('HU19_threshold0.45_min5000_max67000_fullimageverticies_'+version+'.p','rb'))

		g,cl = makeVTK(version,radii=radii,conn=conn,xyz=xyz)
	else:
		cl = read_polydata(file)
	if genGraph:
		graph = generateGraph(cl)
		offset = graph.get_num_of_nodes()
		graph.add_node(offset+0)
		graph.add_node(offset+1)
		graph.add_node(offset+2)
		graph.add_node(offset+3)
		graph.add_outlet(offset+0)
		graph.add_outlet(offset+3)
		graph.add_radius(offset+0,474/2)
		graph.add_radius(offset+1,475/2)
		graph.add_radius(offset+2,477/2)
		graph.add_radius(offset+3,487/2)
		graph.add_edge(offset+0,offset+1,816)
		graph.add_edge(offset+1,offset+2,761)
		graph.add_edge(offset+2,offset+3,403)
		graph.add_edge(offset+2,3830,1130)
		graph.add_edge(offset+2,7371,975)
		graph.add_edge(offset+2,927,923)
		graph.add_edge(offset+2,675,923)

		graph.add_edge(2350,2453,923)
		graph.add_edge(4481,4473,923)
		graph.add_edge(4410,4526,923)
		graph.add_edge(783,728,923)
		graph.add_edge(4644,4807,923)
		graph.add_edge(4246,4403,923)
		graph.add_edge(699,783,923)
		graph.add_edge(2021,2069,923)
		
		
		graph.add_node_xyz(offset+3,[1158.000,441.000,954])
		graph.add_node_xyz(offset+2,[1121.500,278.500,954])
		graph.add_node_xyz(offset+1,[1163.500,137.500,954])
		graph.add_node_xyz(offset+0,[1238.500,35.500,954])


		seed_pt = offset+3
		visited,paths = dijkstra_label(graph,seed_pt)
		g,cl = makeVTK('labeled',graph=graph)

	#print(len(list(graph.edge_radius.keys())))
	junctions,correctedJunctions,vessels,outlets,radius,length,collaterals = writeVesselTreeMetrics_cl(cl)
	removed = None
	iter = 0
	while removed != 0 and iter < 10:
		print(iter)
		
		#print(len(list(removed)))
		iter += 1
	#erodeTerminalBranches_cl(cl,size=2,mode='larger')



	global jointSeg
	global segOrder
	global segDiam
	global segName
	segName = [i for i in range(0,len(list(vessels)))]
	for i in vessels:
		if i not in junctions:
			junctions[i] = []
	jointSeg = junctions
	segOrder = defaultdict(lambda: 0)
	segDiam = radius
	maxOrder = 13
	inletSegName = 0
	[segOrder, maxGen] = initStrahler(segDiam, segOrder, maxOrder, inletSegName, True)
	dill.dump(segOrder,open('segOrder.p','wb'))
	pickle.dump(segDiam,open('segDiam.p','wb'))
	vtk_order = vtk.vtkDoubleArray()
	for branchID in range(0,cl.GetNumberOfCells()):
		vtk_order.InsertNextValue(segOrder[branchID])
	vtk_order.SetName('Order')
	cl.GetCellData().AddArray(vtk_order)

	label_outlier_vessels(cl,segOrder,segDiam,length)

	write_polydata(cl,'graph_final_1.vtp')
	#print(len(list(graph.edge_radius.keys())))
	
	#g,cl = makeVTK('final',graph=graph)
	R_murray = dict()
	outlets.remove(0)
	outlets.remove(6)

	cor_outlets = outlets
	for i in cor_outlets:
		R_murray[i] = radius[i]**3

	Mean_pressure = 0
	Mean_flow = 1667

	pConv = 133.33333333

	R_tot = Mean_pressure*pConv/Mean_flow

	Res_i = dict()

	for i in R_murray:
		RA = np.sum(list(R_murray))*R_tot
		Res_i[i] = RA/R_murray[i]

	pickle.dump(Res_i,open('resistance.p','wb'))







	

if __name__ == '__main__':
	main()