# encoding: utf-8
# Broadcast protocol used to build a distributed cluster of solver processes.
#
# This code uses the twisted event driven networking framework:
#    http://twistedmatrix.com/documents/13.0.0/core/howto/servers.html
#
import json
import logging
import threading
import time
import twisted
from twisted.internet.endpoints import clientFromString
from twisted.internet import reactor
from twisted.internet.protocol import Factory
from twisted.internet.protocol import ReconnectingClientFactory
from twisted.protocols.basic import LineReceiver

__author__ = 'Jason Ansel'

log = logging.getLogger(__name__)

# Simple shared secret to identify our peers
PROTO_PASSWORD = 'ooLeel9aiJ4iW1nei1sa8Haichaig2ch'


class GlobalBest(object):
    """
    Singleton class used to store the global best.  Lock is required in
    the worker processes as networking code runs in a different thread than
    the solver.
    """
    puzzle_id = None
    score = None
    solution = None
    timestamp = time.time()
    lock = threading.Lock()

    @classmethod
    def reset(cls, puzzle_id=None):
        """
        Called when starting a new puzzle

        :param puzzle_id: string identifying the puzzle being solved
        """
        with cls.lock:
            cls.puzzle_id = puzzle_id
            cls.score = None
            cls.solution = None
            cls.timestamp = time.time()

    @classmethod
    def update(cls, puzzle_id, score, solution):
        """
        Replace the current global best if score is lower than GlobalBest.score

        :param puzzle_id: string identifying the puzzle being solved
        :param score: number of squares required by solution
        :param solution: packed permutation representation
        :return: True if a new global best was established
        """
        with cls.lock:
            if puzzle_id != cls.puzzle_id:
                log.warning('discarding results for wrong puzzle %s != %s',
                            puzzle_id, cls.puzzle_id)
            elif cls.score is None or score < cls.score:
                cls.score = score
                cls.solution = solution
                cls.timestamp = time.time()
                return True
            return False


class GrideaProtocol(LineReceiver):
    """
    Network protocol used to communicate problem instances and solutions
    to other processes.  All messages are broadcast to entire network and
    consist of a JSON string on a single line.  There exist two message types:

    1) Problem instances, cause workers to start solving:
    {'id': string, 'puzzle': [[...], ...], ...}

    2) New global best solutions, sent by workers:
    {'puzzle_id': string,  'score': int,  'solution': [...]}
    """
    peers = set()  # GrideaProtocol() instances
    broadcast_lock = threading.Lock()

    def __init__(self, worker=None, on_connect=None):
        """
        :param worker: optional instance of gridea.GrideaWorker()
        :param on_connect: optional callback for after connection
        """
        self.worker = worker
        self.on_connect = on_connect
        self.logged_in = False

    def connectionMade(self):
        """
        Called by twisted framework on connect.
        """
        self.transport.setTcpKeepAlive(True)
        self.transport.setTcpNoDelay(True)
        if isinstance(self.transport, twisted.internet.tcp.Client):
            self.sendLine(PROTO_PASSWORD)
            self.logged_in = True
            GrideaProtocol.peers.add(self)
        log.info('connect (%d peers)', len(GrideaProtocol.peers))
        if self.on_connect:
            self.on_connect()

    def connectionLost(self, reason=None):
        """
        Called by twisted framework on disconnect.
        """
        GrideaProtocol.peers.discard(self)
        log.info('disconnect (%d peers)', len(GrideaProtocol.peers))
        if (isinstance(self.transport, twisted.internet.tcp.Client) and
                reactor.running):
            log.info('shutting down')
            reactor.stop()

    def lineReceived(self, line):
        """
        Called by twisted framework from incoming network messages.

        :param line: the line received from the network
        """
        if not self.logged_in:
            return self.login(line)

        msg = json.loads(line)
        if 'puzzle' in msg:
            # Start solving a new puzzle instance
            GlobalBest.reset(msg['id'])
            if self.worker:
                reactor.callInThread(self.worker.solve, msg)
            self.broadcast(line)
            log.debug('got new puzzle %s', msg['id'])
        elif 'score' in msg:
            # A new global best was found by other process
            self.best(msg['puzzle_id'], msg['score'], msg['solution'])

    def login(self, password):
        """
        Called for any message sent by a client not logged in.  We use a
        simple shared secret auth to make sure we are talking to others who
        speak the same protocol.

        :param password: the message from the client
        """
        if password == PROTO_PASSWORD:
            self.logged_in = True
            GrideaProtocol.peers.add(self)
            log.info('login ok (%d peers)', len(GrideaProtocol.peers))
        else:
            self.transport.loseConnection()
            log.info('login failed (%d peers)', len(GrideaProtocol.peers))

    def broadcast(self, line):
        """
        Broadcast line to all connected peers.  Broadcast lock is only required
        in worker processes as the solver will send from another thread.

        :param line: the line to broadcast
        """
        with GrideaProtocol.broadcast_lock:
            for peer in GrideaProtocol.peers:
                if peer is not self:
                    peer.sendLine(line)

    def best(self, puzzle_id, score, solution):
        """
        Record a new solution to the puzzle, and broadcast it to other
        processes if it is a new global best.

        :param puzzle_id: string identifying the puzzle being solved
        :param score: number of squares required by solution
        :param solution: packed permutation representation
        """
        if GlobalBest.update(puzzle_id, score, solution):
            self.broadcast(json.dumps({'puzzle_id': puzzle_id, 'score': score,
                                       'solution': solution}))


def listen(port):
    """
    Start a server using GrideaProtocol

    :param port: port to listen on
    """
    class ServerFactory(Factory):
        protocol = GrideaProtocol
    reactor.listenTCP(port, ServerFactory())


def connect(hostname, worker=None, on_connect=None):
    """
    Connect to server using GrideaProtocol, automatically retry if it is
    not yet running.

    :param hostname: `hostname:port` to connect to
    :param worker: optional gridea.GrideaWorker() to make this process a worker
    :param on_connect: optional callback after connection
    """
    class ClientFactory(ReconnectingClientFactory):
        def buildProtocol(self, addr):
            return GrideaProtocol(worker, on_connect)
    clientFromString(reactor, 'tcp:' + hostname).connect(ClientFactory())
