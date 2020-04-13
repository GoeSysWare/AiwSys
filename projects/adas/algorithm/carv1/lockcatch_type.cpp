#include "lockcatch_type.h"

namespace watrix {
	namespace algorithm {

		bool LockcatchType::lockcatch_status_t::equals(const LockcatchType::lockcatch_status_t& other)
		{
			// a.equals(b)  a must not be const object,other error occur.
			return (this->left_luomao_status == other.left_luomao_status) &&
				(this->left_tiesi_status == other.left_tiesi_status) &&
				(this->right_luomao_status == other.right_luomao_status) &&
				(this->right_tiesi_status == other.right_tiesi_status);
		}

	}
}// end namespace

