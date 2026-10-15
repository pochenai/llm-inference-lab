use crate::types::RequestId;

#[derive(Debug)]
pub struct ModelOutput {
    pub request_id: RequestId,
    pub token_ids: Vec<u32>,
}
