CREATE TABLE IF NOT EXISTS field_mapping (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  sourcetype VARCHAR(128) NOT NULL,
  source_field VARCHAR(255) NOT NULL,
  mapping_type ENUM('ecs','non-ecs') NOT NULL,
  mapped_field_name VARCHAR(255) NOT NULL,
  ecs_version VARCHAR(32) DEFAULT '8.10.0',
  rationale TEXT NULL,
  confidence DECIMAL(5,2) DEFAULT 0.80,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uniq_src (sourcetype, source_field, mapped_field_name),
  INDEX idx_sourcetype (sourcetype),
  INDEX idx_source_field (source_field)
);

-- Example facts you mentioned:
INSERT INTO field_mapping (sourcetype, source_field, mapping_type, mapped_field_name, rationale, confidence)
VALUES
  ('pan_traffic', 'department_name', 'non-ecs', 'custom_prefix_department_name',
   'Field not in ECS; kept with custom_ prefix for downstream analytics.', 0.95);

-- A few more realistic seeds for similarity testing
INSERT INTO field_mapping (sourcetype, source_field, mapping_type, mapped_field_name, rationale, confidence) VALUES
('pan_traffic', 'department',       'non-ecs', 'custom_prefix_department',       'Alias of department_name observed in traffic logs.', 0.90),
('pan_traffic', 'dept',             'non-ecs', 'custom_prefix_dept',             'Abbreviated department field.', 0.88),
('pan_traffic', 'department_addr',  'non-ecs', 'custom_prefix_department_addr',  'Address form for department.', 0.85),
('pan_traffic', 'dept_address',     'non-ecs', 'custom_prefix_dept_address',     'Department postal address.', 0.86),
('pan_threat',  'src_user_dept',    'non-ecs', 'custom_prefix_src_user_dept',    'Threat context user department.', 0.82),
('pan_threat',  'department_name',  'non-ecs', 'custom_prefix_department_name',  'Threat events carry same concept as traffic.', 0.90),
('pan_threat',  'department',       'non-ecs', 'custom_prefix_department',       'Threat events alias.', 0.85);
