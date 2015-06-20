# encoding: utf-8
# Interact with the challenge API server and submit solutions
import argparse
import json
import logging
import math
import network
import numpy as np
import requests
import scoring
import time
from twisted.internet import reactor

__author__ = 'Jason Ansel'

log = logging.getLogger(__name__)


class ChallengeAPI(object):
    """
    Interface for communicating with the challenge API server
    """
    BASE_URL = 'http://techchallenge.cimpress.com'

    def __init__(self, key, mode):
        """
        Constructor

        :param key: API key from registration site
        :param mode: either 'trial' or 'contest'
        """
        self.key = key
        self.mode = mode
        assert key
        assert mode in ('trial', 'contest')

    def get(self):
        """
        Retrieve a new puzzle instance.

        :return: dictionary with keys 'id', 'puzzle', 'height', 'width'
        """
        url = '{0}/{1}/{2}/puzzle'.format(self.BASE_URL, self.key, self.mode)
        return json.loads(requests.get(url).text)

    def submit(self, puzzle_id, squares):
        """
        Submit results for a puzzle returned by self.get().

        :param puzzle_id: puzzle['id'] from get()
        :param squares: list of {'X': int, 'Y': int, 'Size': int}
        :return dictionary with keys 'score', 'timePenalty', 'errors'
        """
        url = '{0}/{1}/{2}/solution'.format(self.BASE_URL, self.key, self.mode)
        solution = {'id': puzzle_id, 'squares': squares}
        return json.loads(requests.post(url, data=json.dumps(solution)).text)


class LocalTestingAPI(object):
    """
    Testing interface with same signature as ChallengeAPI(), returns a constant
    file without talking to the server.  Used for testing on known puzzles.
    """
    def __init__(self, filename):
        self.filename = filename

    def get(self):
        return json.load(open(self.filename))

    def submit(self, puzzle_id, squares):
        return {'score': len(squares), 'timePenalty': 0, 'errors': []}


class GrideaSubmitClient(object):
    """
    Request puzzles from the challenge API, submit them to the network of
    processes to solve, then submit the best solutions after args.limit
    seconds.
    """

    def __init__(self, args):
        """
        Constructor.

        :param args: namespace produced by argparse
        """
        self.args = args
        self.count = 0
        self.start = 0
        self.puzzle = None
        self.scores_log = []
        self.timings_log = []
        if args.mode == 'local':
            self.api = LocalTestingAPI(args.filename)
        else:
            self.api = ChallengeAPI(args.key, args.mode)
        self.network = network.GrideaProtocol()

    def start_next_puzzle(self):
        """
        Request a puzzle from the challenge API and post it to our network
        of worker processes.
        """
        self.count += 1
        self.start = time.time()
        self.puzzle = self.api.get()
        network.GlobalBest.reset(self.puzzle['id'])
        self.network.broadcast(json.dumps(self.puzzle))
        # Schedule submission self.args.limit seconds after self.start
        reactor.callLater(self.start - time.time() + self.args.limit,
                          self.submit_results)

    def submit_results(self):
        """
        Submit the global best result to the challenge API.  Print out how
        well we did.
        """
        result = scoring.expand_solution(np.array(self.puzzle['puzzle'],
                                                  np.uint8),
                                         network.GlobalBest.solution)
        response = self.api.submit(self.puzzle['id'], result)
        assert len(response['errors']) == 0
        print('{:6.3f}: {:3} OK {:3} + {:3}: {}x{} {:.3f}'.format(
            time.time() - self.start, self.count, response['score'],
            response['timePenalty'], self.puzzle.get('height'),
            self.puzzle.get('width'),
            network.GlobalBest.timestamp - self.start))

        # Log how we did for final printout
        self.scores_log.append(network.GlobalBest.score)
        self.timings_log.append(network.GlobalBest.timestamp - self.start)

        if self.count < self.args.count:
            # Wait two seconds for things to cool down, then repeat
            reactor.callLater(2, self.start_next_puzzle)
        else:
            # Print stats and shutdown
            # 90% confidence in standard error of mean wki.pe/Standard_error
            confidence_of_mean = 1.96 / math.sqrt(len(self.scores_log))
            print('mean score {:.1f} +- {:.1f}, mean timing {:.2f} +- {:.2f}'
                  .format(np.mean(self.scores_log),
                          np.std(self.scores_log) * confidence_of_mean,
                          np.mean(self.timings_log),
                          np.std(self.timings_log) * confidence_of_mean))
            reactor.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', help='key for challenge API server')
    parser.add_argument('--mode', default='local',
                        choices=('local', 'trial', 'contest'),
                        help='environment to report to challenge API')
    parser.add_argument('--limit', default=9.6, type=float,
                        help='seconds to wait before reporting results')
    parser.add_argument('--hostname', default='localhost:8099',
                        help='worker process cluster hostname:port connect to')
    parser.add_argument('--filename', default='example_puzzle.json',
                        help='puzzle JSON file for --mode=local')
    parser.add_argument('--count', '-n', default=1, type=int, metavar='N',
                        help='count of number of puzzles to solve')
    parser.add_argument('--debug', '-v', action='store_true',
                        help='print verbose debugging output')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)
    network.connect(args.hostname,
                    on_connect=GrideaSubmitClient(args).start_next_puzzle)
    reactor.run()


if __name__ == '__main__':
    main()
