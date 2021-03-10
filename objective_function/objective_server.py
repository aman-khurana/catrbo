from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
from objective import eval_objective
import torch

PORT = 8000

# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

# Create server
with SimpleXMLRPCServer(('0.0.0.0', PORT),
                        requestHandler=RequestHandler) as server:
    server.register_introspection_functions()

    
    def evaluate(x, hyperparams, lb, ub, time_precision = 2):
        
        x = torch.tensor(x).to(dtype = torch.float, 
                                        device = torch.device("cpu"))

        f1, time = eval_objective(x, hyperparams, lb, ub, time_precision = 2)

        f1 = float(f1)
        time = float(time)

        return f1, time

    server.register_function(evaluate, 'evaluate')

    # Run the server's main loop
    server.serve_forever()

