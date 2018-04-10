# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import communication.grpc_pb2 as grpc__pb2


class ServerMessagesStreamStub(object):
    # missing associated documentation comment in .proto file
    pass

    def __init__(self, channel):
        """Constructor.

        Args:
          channel: A grpc.Channel.
        """
        self.registerListener = channel.unary_stream(
            '/ServerMessagesStream/registerListener',
            request_serializer=grpc__pb2.Booly.SerializeToString,
            response_deserializer=grpc__pb2.XmlMessage.FromString,
        )
        self.registerEventSource = channel.stream_unary(
            '/ServerMessagesStream/registerEventSource',
            request_serializer=grpc__pb2.XmlMessage.SerializeToString,
            response_deserializer=grpc__pb2.Booly.FromString,
        )


class ServerMessagesStreamServicer(object):
    # missing associated documentation comment in .proto file
    pass

    def registerListener(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def registerEventSource(self, request_iterator, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ServerMessagesStreamServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'registerListener': grpc.unary_stream_rpc_method_handler(
            servicer.registerListener,
            request_deserializer=grpc__pb2.Booly.FromString,
            response_serializer=grpc__pb2.XmlMessage.SerializeToString,
        ),
        'registerEventSource': grpc.stream_unary_rpc_method_handler(
            servicer.registerEventSource,
            request_deserializer=grpc__pb2.XmlMessage.FromString,
            response_serializer=grpc__pb2.Booly.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'ServerMessagesStream', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


class ControlEventHandlerStub(object):
    # missing associated documentation comment in .proto file
    pass

    def __init__(self, channel):
        """Constructor.

        Args:
          channel: A grpc.Channel.
        """
        self.submitControlEvent = channel.unary_unary(
            '/ControlEventHandler/submitControlEvent',
            request_serializer=grpc__pb2.ControlEvent.SerializeToString,
            response_deserializer=grpc__pb2.Booly.FromString,
        )


class ControlEventHandlerServicer(object):
    # missing associated documentation comment in .proto file
    pass

    def submitControlEvent(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ControlEventHandlerServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'submitControlEvent': grpc.unary_unary_rpc_method_handler(
            servicer.submitControlEvent,
            request_deserializer=grpc__pb2.ControlEvent.FromString,
            response_serializer=grpc__pb2.Booly.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'ControlEventHandler', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))