"""
Function Call Orchestrator Service
Handles routing of Gemini function calls to actual API endpoints
"""

import logging
from typing import Dict, Any, Optional
from app.services.appointment_service import appointment_service

logger = logging.getLogger(__name__)


class FunctionOrchestrator:
    """Orchestrates function calls from Gemini to actual API endpoints"""

    def __init__(self):
        self.appointment_service = appointment_service

    async def execute_function(
        self,
        function_name: str,
        function_args: Dict[str, Any],
        customer_id: Optional[int] = None,
        customer_email: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a function call from Gemini

        Args:
            function_name: Name of the function to call
            function_args: Arguments for the function
            customer_id: Customer ID
            customer_email: Customer email
            auth_token: JWT authentication token

        Returns:
            Function execution result
        """
        logger.info(f"Executing function: {function_name} with args: {function_args}")

        try:
            if function_name == "get_user_appointments":
                return await self._get_user_appointments(
                    function_args, customer_id, customer_email, auth_token
                )

            elif function_name == "get_user_vehicles":
                return await self._get_user_vehicles(auth_token)

            elif function_name == "book_appointment":
                return await self._book_appointment(
                    function_args, auth_token
                )

            else:
                logger.error(f"Unknown function: {function_name}")
                return {
                    "success": False,
                    "error": f"Unknown function: {function_name}"
                }

        except Exception as e:
            logger.error(f"Error executing function {function_name}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    async def _get_user_appointments(
        self,
        args: Dict[str, Any],
        customer_id: Optional[int],
        customer_email: Optional[str],
        auth_token: Optional[str]
    ) -> Dict[str, Any]:
        """Get user appointments"""
        appointment_type = args.get("appointment_type", "all")

        appointments = await self.appointment_service.get_appointments_for_customer(
            customer_id=customer_id,
            customer_email=customer_email,
            appointment_type=appointment_type,
            auth_token=auth_token
        )

        if not appointments:
            return {
                "success": True,
                "data": [],
                "message": "No appointments found"
            }

        return {
            "success": True,
            "data": appointments,
            "count": len(appointments)
        }

    async def _get_user_vehicles(
        self,
        auth_token: Optional[str]
    ) -> Dict[str, Any]:
        """Get user vehicles"""
        vehicles = await self.appointment_service.get_customer_vehicles(
            auth_token=auth_token
        )

        if not vehicles:
            return {
                "success": True,
                "data": [],
                "message": "No vehicles found. Please add a vehicle first."
            }

        return {
            "success": True,
            "data": vehicles,
            "count": len(vehicles)
        }

    async def _book_appointment(
        self,
        args: Dict[str, Any],
        auth_token: Optional[str]
    ) -> Dict[str, Any]:
        """Book an appointment"""
        # Validate required fields
        required_fields = ["vehicle_id", "appointment_date", "start_time", "consultation_type", "customer_issue"]
        missing_fields = [field for field in required_fields if field not in args]

        if missing_fields:
            return {
                "success": False,
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }

        # Validate vehicle exists for this user
        try:
            vehicle_id = int(args["vehicle_id"])

            # Get user's vehicles to validate
            user_vehicles = await self.appointment_service.get_customer_vehicles(auth_token=auth_token)

            if not user_vehicles:
                return {
                    "success": False,
                    "error": "No vehicles found. Please add a vehicle before booking an appointment."
                }

            # Check if vehicle_id exists in user's vehicles
            vehicle_ids = [v.get("id") for v in user_vehicles]

            if vehicle_id not in vehicle_ids:
                vehicle_list = ", ".join([
                    f"{v.get('year', '')} {v.get('make')} {v.get('model')} (ID: {v.get('id')})"
                    for v in user_vehicles
                ])
                return {
                    "success": False,
                    "error": f"Invalid vehicle ID {vehicle_id}. Your vehicles: {vehicle_list}"
                }

        except (ValueError, TypeError) as e:
            return {
                "success": False,
                "error": f"Invalid vehicle_id: {args.get('vehicle_id')}"
            }

        # Call booking API
        result = await self.appointment_service.book_appointment(
            vehicle_id=vehicle_id,
            appointment_date=args["appointment_date"],
            start_time=args["start_time"],
            consultation_type=args["consultation_type"],
            customer_issue=args["customer_issue"],
            auth_token=auth_token
        )

        # Check for errors
        if "error" in result:
            return {
                "success": False,
                "error": result["error"]
            }

        return {
            "success": True,
            "data": result,
            "message": "Appointment booked successfully"
        }


# Global instance
function_orchestrator = FunctionOrchestrator()
