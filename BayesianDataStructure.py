import collections
import math
import csv
import copy
import logging
import datetime

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class Node:
    def __init__(self, NodeName, TotalMeasurement, Colvals, Measurements):
        self.NodeName = NodeName
        self.TotalMeasurement = TotalMeasurement
        self.OriginalVals = Colvals

        self.PDistribution = collections.defaultdict(float)
        for i, val in enumerate(Colvals):
            self.PDistribution[val] += Measurements[i]

        for key in self.PDistribution:
            self.PDistribution[key] /= self.TotalMeasurement

        self.UniqueVals = self.PDistribution.keys()

    def __str__(self):
        return "%s %s %s" % (self.NodeName, len(self.UniqueVals), str(self.PDistribution))

class Edge:
    def __init__(self, node1, node2, CMI):
        self.Nodes = [node1, node2]
        self.CMI = CMI
        #-1, 0, 1
        self.Direction = 0

    def __str__(self):
        DirectionSymbols = ("-", "->", "<-")
        linkSymbol = DirectionSymbols[self.Direction]
        ret = "%s%s%s: %s" % (self.Nodes[0].NodeName, linkSymbol[self.Direction], self.Nodes[1].NodeName, self.CMI)
        return ret

class Graph:
    def __init__(self, NodeNumber, NodesList):
        self.Connections = [[None for j in xrange(NodeNumber)] for i in xrange(NodeNumber)]
        self.Directions = [[None for j in xrange(NodeNumber)] for i in xrange(NodeNumber)]
        self.NodesList = NodesList
        self.NodeNumber = len(self.NodesList)
        self.NodesDict = dict(zip(NodesList, [i for i in xrange(len(NodesList))] ))
        self.EdgeNumber = 0
        self.Edges = []
        self.DrawFunc = None

    def __str__(self):
        ret = ""
        for line in self.Connections:
            ret += str(line) + "\n"
        return ret

    def DrawGraphDirected(self):
        if self.DrawFunc is None:
            raise Exception("no draw function assigned yet")
        graph, CMIs = self.GetDirectedEdgesAndCMIs()
        CMIs4f = map(lambda x:"%.4f"%x, CMIs)
        self.DrawFunc(graph, CMIs4f, True)

    def DrawGraph(self):
        if self.DrawFunc is None:
            raise Exception("no draw function assigned yet")
        graph, CMIs = self.GetEdgesAndCMIs()
        CMIs4f = map(lambda x:"%.4f"%x, CMIs)
        self.DrawFunc(graph, CMIs4f, False)

    def addEdge(self, edge):
        self.EdgeNumber += 1
        node1Seq = self.NodesDict[edge.Nodes[0]]
        node2Seq = self.NodesDict[edge.Nodes[1]]
        self.Connections[node1Seq][node2Seq] = edge.CMI
        self.Connections[node2Seq][node1Seq] = edge.CMI

    def delEdge(self, node1, node2):
        node1Seq = self.NodesDict[node1]
        node2Seq = self.NodesDict[node2]
        edge = Edge(node1, node2, self.Connections[node1Seq][node2Seq])
        self.Connections[node1Seq][node2Seq] = None
        return edge

    def twoNodeConnected(self, node1, node2):
        node1Seq = self.NodesDict[node1]
        node2Seq = self.NodesDict[node2]
        viewmap = [[False for j in xrange(self.NodeNumber)] for i in xrange(self.NodeNumber)]
        return self.DFSNodeConnected(node1Seq, node2Seq, viewmap)

    def DFSNodeConnected(self, node1Seq, node2Seq, viewmap):
        for i, directConnectNode in enumerate(self.Connections[node1Seq]):
            #node1Seq and i are connected
            if directConnectNode is not None:
                if i == node2Seq:
                    return True
                else:
                    if not viewmap[node1Seq][i]:
                        viewmap[node1Seq][i] = True
                        conn = self.DFSNodeConnected(i, node2Seq, viewmap)
                        if conn: return conn
        return False

    def GetNeighbors(self, node):
        ret = set()
        nodeSeq = self.NodesDict[node]
        for i, CMI in enumerate(self.Connections[nodeSeq]):
            if CMI is not None:
                ret.add(self.NodesList[i])

        for j, line in enumerate(self.Connections):
            if line[nodeSeq] is not None:
                ret.add(self.NodesList[j])

        return ret

    def GetChildren(self, node):
        ret = set()
        nodeSeq = self.NodesDict[node]
        for i, CMI in enumerate(self.Connections[nodeSeq]):
            if CMI is not None:
                if self.Connections[i][nodeSeq] is None:
                    ret.add(self.NodesList[i])
        return ret

    def GetNeighborsWithoutChildren(self, node):
        Nb = self.GetNeighbors(node)
        chd = self.GetChildren(node)
        ret = Nb - chd
        return ret

    def GetEdgesAndCMIs(self):
        ret = []
        CMIs = []
        for i, line in enumerate(self.Connections):
            for j, val in enumerate(line):
                if val is not None:
                    if i<=j:
                        ret.append((self.NodesList[i].NodeName, self.NodesList[j].NodeName))
                        CMIs.append(val)
        return ret, CMIs

    def GetDirectedEdgesAndCMIs(self):
        ret = []
        CMIs = []
        for i, line in enumerate(self.Directions):
            for j, val in enumerate(line):
                if val is not None:
                    ret.append((self.NodesList[i].NodeName, self.NodesList[j].NodeName))
                    CMIs.append(self.Connections[i][j])
        return ret, CMIs

    def GetPathBetweenTwoNodes(self, node1, node2):
        paths = self.GetPathHelper(node1, node2, self.Connections)
        return paths

    def GetDirectionPathBetweenTwoNodes(self, node1, node2):
        paths = self.GetPathHelper(node1, node2, self.Directions)
        return paths

    def GetPathHelper(self, node1, node2, Connections):
        node1Seq = self.NodesDict[node1]
        node2Seq = self.NodesDict[node2]
        paths = []
        nowpath = []
        bannedNodeSeqSet = set()
        self.GetPathDFS(node1Seq, node2Seq, nowpath, paths, bannedNodeSeqSet, Connections)
        return paths

    def GetPathDFS(self, node1Seq, node2Seq, nowpath, paths, bannedNodeSeqSet, Connections):
        #logger.debug(nowpath)
        nowpath.append(node1Seq)
        bannedNodeSeqSet.add(node1Seq)

        if node1Seq == node2Seq:
            paths.append(copy.copy(nowpath))
        
        for i, neibor in enumerate(Connections[node1Seq]):
            if neibor is not None and i not in bannedNodeSeqSet:
                self.GetPathDFS(i, node2Seq, nowpath, paths, bannedNodeSeqSet, Connections)

        bannedNodeSeqSet.remove(node1Seq)
        nowpath.pop()

    def GetNeighborsInPathBetweenTwoNodes(self, node1, node2):
        paths = self.GetPathBetweenTwoNodes(node1, node2)
        neiborSet1 = set()
        neiborSet2 = set()
        for path in paths:
            if len(path) >= 3:
                neiborSet1.add(self.NodesList[path[1]])
                neiborSet2.add(self.NodesList[path[-2]])
        return neiborSet1, neiborSet2

    def SetEdgeDirection(self, node1, node2):
        node1Seq = self.NodesDict[node1]
        node2Seq = self.NodesDict[node2]

        if self.Directions[node2Seq][node1Seq]:
            logger.debug("[linked reverse direction] %s->%s" % (node2.NodeName, node1.NodeName))
            return
            raise Exception("[linked] %s->%s" % (node2.NodeName, node1.NodeName))

        if self.Directions[node1Seq][node2Seq] is not True:
            logger.debug("edge direction %s->%s" % (node1.NodeName, node2.NodeName))
            self.Directions[node1Seq][node2Seq] = True
        else:
             logger.debug("[linked] %s->%s" % (node1.NodeName, node2.NodeName))

    def SetEdgeDirectionForce(self, node1, node2):
        node1Seq = self.NodesDict[node1]
        node2Seq = self.NodesDict[node2]

        self.Directions[node2Seq][node1Seq] = None

        if self.Directions[node1Seq][node2Seq] is not True:
            self.Directions[node1Seq][node2Seq] = True

    def ParentOf(self, node1, node2):
        node1Seq = self.NodesDict[node1]
        node2Seq = self.NodesDict[node2]
        if self.Directions[node1Seq][node2Seq]:
            return True
        else:
            return False

    def GetParents(self, node):
        ParentSet = set()
        nodeSeq = self.NodesDict[node]
        for i in xrange(len(self.NodesList)):
            if self.Directions[i][nodeSeq] is not None:
                ParentSet.add(self.NodesList[i])
        return ParentSet

    def Adjacent(self, node1, node2):
        node1Seq = self.NodesDict[node1]
        node2Seq = self.NodesDict[node2]
        if self.Connections[node1Seq][node2Seq] is not None:
            return True
        else:
            return False

class BayesianNetwork:
    def __init__(self, ColumnNames, ColumnVals, Measurements):
        self.ColumnNames = ColumnNames
        self.Measurements = Measurements
        self.TotalMeasurement = sum(self.Measurements)
        self.Nodes = []
        self.NodesDict = {}
        self.AllEdges = []
        self.epsilon = 0.003

        for i, columnName in enumerate(ColumnNames):
            onenode = Node(columnName, self.TotalMeasurement, ColumnVals[i], Measurements)
            self.NodesDict[columnName] = onenode
            self.Nodes.append(onenode)

        self.BasicGraph = Graph(len(self.Nodes), self.Nodes)

    def Draft(self):
        nodesize = len(self.Nodes)
        logger.debug("total edges:" + str(nodesize*(nodesize-1)/2))
        for i in xrange(nodesize-1):
            for j in xrange(i+1, nodesize):
                CMI = self.ConditionalMutualInformation(self.Nodes[i], self.Nodes[j], [])
                edge = Edge(self.Nodes[i], self.Nodes[j], CMI)
                if CMI >= self.epsilon:
                    self.AllEdges.append(edge)

        #sort
        self.AllEdges.sort(cmp=lambda x,y: -1 if y.CMI - x.CMI > 0 else 1)
        for edge in self.AllEdges:
            logger.debug(str(edge))
        logger.debug("all valid edges:" + str(len(self.AllEdges)))

        while self.AllEdges:
            if self.BasicGraph.EdgeNumber == nodesize-1:
                break

            biggestEdge = self.AllEdges.pop()
            if not self.BasicGraph.twoNodeConnected(*(biggestEdge.Nodes)):
                self.BasicGraph.addEdge(biggestEdge)
        logger.debug("Graph ends, edges left:%d" % len(self.AllEdges))
        self.BasicGraph.DrawGraph()

    def Thickening(self):
        while self.AllEdges:
            edge = self.AllEdges.pop()
            sepRes = None
            sepRes = self.TryToSeperate(self.BasicGraph, edge.Nodes[0], edge.Nodes[1])
            if not sepRes:
                logger.debug("[not seperate] %s %s" % (edge.Nodes[0].NodeName, edge.Nodes[1].NodeName))
                self.BasicGraph.addEdge(edge)
            else:
                logger.debug("[seperated] %s %s" % (edge.Nodes[0].NodeName, edge.Nodes[1].NodeName))

    def Thinning(self):
        for edge in self.BasicGraph.Edges:
            paths = self.BasicGraph.GetPathBetweenTwoNodes(*(edge.Nodes))
            if len(paths) >= 2:
                edgeDeletedTmp = self.BasicGraph.delEdge(*(edge.Nodes))
                separable = self.TryToSeperate(self.BasicGraph, *(edge.Nodes))
                if not separable:
                    self.BasicGraph.addEdge(edgeDeletedTmp)

    def ConditionalMutualInformation(self, Xi, Xj, Conditions, SpecificConditionStrList = None):
        ret = 0
        XiVals = collections.defaultdict(float)
        XjVals = collections.defaultdict(float)
        XiXjVals = collections.defaultdict(float)

        ConditionStr = None
        if SpecificConditionStrList:
            ConditionStr = ",".join(SpecificConditionStrList)

        TotalLineNumber = len(Xi.OriginalVals)
        for i in xrange(TotalLineNumber):
            thekey = ""

            if Conditions:
                thekey = ",".join([condition.OriginalVals[i] for condition in Conditions])
                #not the specific condition
                if ConditionStr:
                    if thekey != ConditionStr:
                        continue

            XiVals[(thekey, Xi.OriginalVals[i])] += self.Measurements[i]
            XjVals[(thekey, Xj.OriginalVals[i])] += self.Measurements[i]
            XiXjVals[(thekey, Xi.OriginalVals[i], Xj.OriginalVals[i])] += self.Measurements[i]

        #change number to possibility
        for key in XiVals:
            XiVals[key] /= self.TotalMeasurement
        for key in XjVals:
            XjVals[key] /= self.TotalMeasurement
        for key in XiXjVals:
            XiXjVals[key] /= self.TotalMeasurement

        #calculate mutual information
        for xixjKey in XiXjVals:
            xiKey = (xixjKey[0], xixjKey[1])
            xjKey = (xixjKey[0], xixjKey[2])

            Pcxi = XiVals[xiKey]
            Pcxj = XjVals[xjKey]
            Pcxixj = XiXjVals[xixjKey]

            thismi = Pcxixj * math.log(Pcxixj/(Pcxi*Pcxj))
            ret += thismi

        return ret

    def TryToSeperate(self, CurrentGraph, node1, node2, StaticCondition = False):
        time1 = datetime.datetime.now()
        N1, N2 = self.BasicGraph.GetNeighborsInPathBetweenTwoNodes(node1, node2)
        N1Children = self.BasicGraph.GetChildren(node1)
        N2Children = self.BasicGraph.GetChildren(node2)
        N1 -= N1Children
        N2 -= N2Children

        if len(N1) > len(N2):
            tmp = N1; N1 = N2; N2 = tmp

        N = [N1, N2]
        if N1 == N2:
            N = [N1]
        for i in xrange(len(N)):
            conditionset = N[i]
            logger.debug("[try to seperate] %s-%s on conditions %s" % (node1.NodeName, node2.NodeName, ",".join(map(lambda x: x.NodeName, conditionset))))
            conditionlist = list(conditionset)
            StaticConditionlist = None
            if StaticCondition:
                pass
            v = self.ConditionalMutualInformation(node1, node2, conditionlist, StaticConditionlist)
            if v < self.epsilon:
                return True

            while True:
                #STEP 6
                if len(conditionlist) < 2:
                    break

                Cm = None
                Vm = None
                for i in xrange(len(conditionlist)):
                    newconditionlist = conditionlist[:i] + conditionlist[i+1:]
                    StaticConditionlist = None
                    if StaticCondition:
                        pass
                    vt = self.ConditionalMutualInformation(node1, node2, list(newconditionlist), StaticConditionlist)
                    if vt < self.epsilon:
                        return True

                    if Vm is None or vt < Vm:
                        Vm = vt
                        Cm = newconditionlist

                #STEP 7
                #have found the smallest vm
                if Vm < self.epsilon:
                    return True
                elif Vm > v:
                    break
                else:
                    v = Vm
                    conditionlist = Cm

        return False

    def GetStaticConditionList(self):
        pass

    def OrientEdge(self, CurrentGraph = None):
        if CurrentGraph is None:
            CurrentGraph = self.BasicGraph
        undecidedTriplet = set()

        EdgedDict = {}
        #STEP1
        for a in self.BasicGraph.NodesList:
            nodeNeiborsSet = self.BasicGraph.GetNeighbors(a)
            Sa = list(nodeNeiborsSet)
            #get two nodes from a's neighbors
            for i in xrange(len(Sa)):
                for j in xrange(i+1, len(Sa)):
                    s1 = Sa[i]
                    s2 = Sa[j]
                    s1_path_neibors, s2_path_neibors = self.BasicGraph.GetNeighborsInPathBetweenTwoNodes(s1, s2)
                    s1_s2_path_neibors = s1_path_neibors | s2_path_neibors
                    CMI_with_a = self.ConditionalMutualInformation(s1, s2, s1_s2_path_neibors)
                    s1_s2_path_neibors.remove(a)
                    CMI_without_a = self.ConditionalMutualInformation(s1, s2, s1_s2_path_neibors)

                    if CMI_with_a > CMI_without_a:
                        logger.debug("s1:%s a:%s s2:%s %.3f %.3f" % (s1.NodeName, a.NodeName, s2.NodeName, CMI_with_a, CMI_without_a))
                        #let s1 s2 be parent of a
                        CMI_diff = CMI_with_a - CMI_without_a
                        edgeKey1 = frozenset([s1, a])
                        if edgeKey1 not in EdgedDict or (EdgedDict[edgeKey1] < CMI_diff):
                            self.BasicGraph.SetEdgeDirectionForce(s1, a)
                            EdgedDict[edgeKey1] = CMI_diff

                        edgeKey2 = frozenset([s2, a])
                        if edgeKey2 not in EdgedDict or (EdgedDict[edgeKey2] < CMI_diff):
                            self.BasicGraph.SetEdgeDirectionForce(s2, a)
                            EdgedDict[edgeKey2] = CMI_diff

                    elif CMI_with_a == CMI_without_a:
                        undecidedTriplet.add((s1, a, s2))

            if True:
                continue

            for x in Sa:
                if self.BasicGraph.ParentOf(x, a):
                    continue
                #traverse parent of a
                candidates = self.BasicGraph.GetParents(a)
                for parent_of_a in candidates:
                    triplet1= (x, a , parent_of_a)
                    if triplet1 not in undecidedTriplet:
                        #let x be child of a
                        self.BasicGraph.SetEdgeDirection(a, x)

        return
        nodesNum = self.BasicGraph.NodeNumbe
        #orient until no more dege can be orient
        for i in xrange(10):
            #STEP2
            for i in xrange(nodesNum):
                for j in xrange(i+1, nodesNum):
                    for k in xrange(j+1, nodesNum):
                        a, b, c = self.BasicGraph.NodesList[i], self.BasicGraph.NodesList[j], self.BasicGraph.NodesList[k]
                        #if a is parent of b, b and c are adjacent, bc nots oriented and <a,b,c> not undecided.
                        #b be parent of c
                        if self.BasicGraph.ParentOf(a, b) and self.BasicGraph.Adjacent(b, c) and (a, b, c) not in undecidedTriplet:
                            self.BasicGraph.SetEdgeDirection(b, c)

            #STEP3
            undirectedEdges = []
            for ue in undirectedEdges:
                node1 = ue.Nodes[0]
                node2 = ue.Nodes[1]
                #if there is a directed path from node1 to node2, orient the edge node1 -> node2
                dpaths = self.BasicGraph.GetDirectionPathBetweenTwoNodes(node1, node2)
                if dpaths:
                    self.BasicGraph.SetEdgeDirection(node1, node2)


class DataParser:
    def __init__(self):
        self.ColumnNames = []
        self.ColumnVals = []

    def readCsvFile(self, path, hasHeader=True, maxlen=None, Delimiter="\t"):
        ColumnNames = []
        ColumnVals = []
        numline = 0

        with open(path, "rb") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=Delimiter, quotechar='|')
            for row in spamreader:
                #deal header
                numline += 1
                if numline == 1:
                    ColumnVals = [[] for i in xrange(len(row))]
                    if hasHeader:
                        ColumnNames = copy.copy(row)
                        continue
                    else:
                        ColumnNames = ["col"+i for i in xrange(len(row))]

                for i, val in enumerate(row):
                    ColumnVals[i].append(val)

                if maxlen and numline >= maxlen:
                    break

        self.ColumnNames = ColumnNames
        self.ColumnVals = ColumnVals