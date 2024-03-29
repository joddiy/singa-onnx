# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: io.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='io.proto',
  package='singa',
  syntax='proto2',
  serialized_pb=_b('\n\x08io.proto\x12\x05singa\"D\n\x0b\x45ncoderConf\x12\x17\n\x04type\x18\x01 \x01(\t:\tjpg2proto\x12\x1c\n\x0fimage_dim_order\x18\x02 \x01(\t:\x03HWC\"]\n\x0b\x44\x65\x63oderConf\x12\x17\n\x04type\x18\x01 \x01(\t:\tproto2jpg\x12\x1c\n\x0fimage_dim_order\x18\x02 \x01(\t:\x03\x43HW\x12\x17\n\thas_label\x18\x03 \x01(\x08:\x04true\"\x97\x03\n\x0fTransformerConf\x12!\n\x12\x66\x65\x61turewise_center\x18\x01 \x01(\x08:\x05\x66\x61lse\x12 \n\x11samplewise_center\x18\x02 \x01(\x08:\x05\x66\x61lse\x12#\n\x14\x66\x65\x61turewise_std_norm\x18\x03 \x01(\x08:\x05\x66\x61lse\x12\"\n\x13samplewise_std_norm\x18\x04 \x01(\x08:\x05\x66\x61lse\x12\x1c\n\rzca_whitening\x18\x05 \x01(\x08:\x05\x66\x61lse\x12\x19\n\x0erotation_range\x18\x06 \x01(\x05:\x01\x30\x12\x16\n\ncrop_shape\x18\x07 \x03(\rB\x02\x10\x01\x12\x18\n\rresize_height\x18\x08 \x01(\x05:\x01\x30\x12\x17\n\x0cresize_width\x18\t \x01(\x05:\x01\x30\x12 \n\x11horizontal_mirror\x18\n \x01(\x08:\x05\x66\x61lse\x12\x1e\n\x0fvertical_mirror\x18\x0b \x01(\x08:\x05\x66\x61lse\x12\x1c\n\x0fimage_dim_order\x18\x0c \x01(\t:\x03\x43HW\x12\x12\n\x07rescale\x18\r \x01(\x02:\x01\x30\":\n\x0bImageRecord\x12\r\n\x05shape\x18\x01 \x03(\x05\x12\r\n\x05label\x18\x02 \x03(\x05\x12\r\n\x05pixel\x18\x03 \x01(\x0c\x42\x18\n\x16org.apache.singa.proto')
)




_ENCODERCONF = _descriptor.Descriptor(
  name='EncoderConf',
  full_name='singa.EncoderConf',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='singa.EncoderConf.type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("jpg2proto").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='image_dim_order', full_name='singa.EncoderConf.image_dim_order', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("HWC").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=19,
  serialized_end=87,
)


_DECODERCONF = _descriptor.Descriptor(
  name='DecoderConf',
  full_name='singa.DecoderConf',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='singa.DecoderConf.type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("proto2jpg").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='image_dim_order', full_name='singa.DecoderConf.image_dim_order', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("CHW").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='has_label', full_name='singa.DecoderConf.has_label', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=89,
  serialized_end=182,
)


_TRANSFORMERCONF = _descriptor.Descriptor(
  name='TransformerConf',
  full_name='singa.TransformerConf',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='featurewise_center', full_name='singa.TransformerConf.featurewise_center', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='samplewise_center', full_name='singa.TransformerConf.samplewise_center', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='featurewise_std_norm', full_name='singa.TransformerConf.featurewise_std_norm', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='samplewise_std_norm', full_name='singa.TransformerConf.samplewise_std_norm', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='zca_whitening', full_name='singa.TransformerConf.zca_whitening', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='rotation_range', full_name='singa.TransformerConf.rotation_range', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='crop_shape', full_name='singa.TransformerConf.crop_shape', index=6,
      number=7, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
    _descriptor.FieldDescriptor(
      name='resize_height', full_name='singa.TransformerConf.resize_height', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='resize_width', full_name='singa.TransformerConf.resize_width', index=8,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='horizontal_mirror', full_name='singa.TransformerConf.horizontal_mirror', index=9,
      number=10, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='vertical_mirror', full_name='singa.TransformerConf.vertical_mirror', index=10,
      number=11, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='image_dim_order', full_name='singa.TransformerConf.image_dim_order', index=11,
      number=12, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("CHW").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='rescale', full_name='singa.TransformerConf.rescale', index=12,
      number=13, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=185,
  serialized_end=592,
)


_IMAGERECORD = _descriptor.Descriptor(
  name='ImageRecord',
  full_name='singa.ImageRecord',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='shape', full_name='singa.ImageRecord.shape', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='label', full_name='singa.ImageRecord.label', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pixel', full_name='singa.ImageRecord.pixel', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=594,
  serialized_end=652,
)

DESCRIPTOR.message_types_by_name['EncoderConf'] = _ENCODERCONF
DESCRIPTOR.message_types_by_name['DecoderConf'] = _DECODERCONF
DESCRIPTOR.message_types_by_name['TransformerConf'] = _TRANSFORMERCONF
DESCRIPTOR.message_types_by_name['ImageRecord'] = _IMAGERECORD
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

EncoderConf = _reflection.GeneratedProtocolMessageType('EncoderConf', (_message.Message,), dict(
  DESCRIPTOR = _ENCODERCONF,
  __module__ = 'io_pb2'
  # @@protoc_insertion_point(class_scope:singa.EncoderConf)
  ))
_sym_db.RegisterMessage(EncoderConf)

DecoderConf = _reflection.GeneratedProtocolMessageType('DecoderConf', (_message.Message,), dict(
  DESCRIPTOR = _DECODERCONF,
  __module__ = 'io_pb2'
  # @@protoc_insertion_point(class_scope:singa.DecoderConf)
  ))
_sym_db.RegisterMessage(DecoderConf)

TransformerConf = _reflection.GeneratedProtocolMessageType('TransformerConf', (_message.Message,), dict(
  DESCRIPTOR = _TRANSFORMERCONF,
  __module__ = 'io_pb2'
  # @@protoc_insertion_point(class_scope:singa.TransformerConf)
  ))
_sym_db.RegisterMessage(TransformerConf)

ImageRecord = _reflection.GeneratedProtocolMessageType('ImageRecord', (_message.Message,), dict(
  DESCRIPTOR = _IMAGERECORD,
  __module__ = 'io_pb2'
  # @@protoc_insertion_point(class_scope:singa.ImageRecord)
  ))
_sym_db.RegisterMessage(ImageRecord)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\026org.apache.singa.proto'))
_TRANSFORMERCONF.fields_by_name['crop_shape'].has_options = True
_TRANSFORMERCONF.fields_by_name['crop_shape']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
# @@protoc_insertion_point(module_scope)
