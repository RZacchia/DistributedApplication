using BookRent.Orchestrator.Api.Requests;

namespace BookRent.Orchestrator.Services.Interfaces;

public interface IIdentityService
{
    Task<string> LoginRequest(AuthentificationRequest request);
}